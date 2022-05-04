"""Trains a GNN using PyTorchGeometric given a parsed graph dataset.

Returns:
    pkl file: an output model file as well as training results.
"""
# Recollecting with no fixed width (might change everything)
# #####
# 1. Rewrite Neighborhood Sampling in PyTorch Geometric to utilize the GPU fully
# 2. Port GNNVPR to DGL (Sampling, FP16 Models, etc)
# 3. Feature Engineering on VPR (Per-Node History Acceleration & Present Cost Acceleration)
# 4. Feature Engineering 2: Global-Graph Properties (Route Channel Width, Ending Pres Factor, Pres Acceleration)
import ast
import glob
import itertools
import os
import time
from multiprocessing import Pool, cpu_count, freeze_support
from optparse import OptionParser


import networkx as nx
import torch
# import 
import pytorch_lightning as pl
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging
import torch.nn.functional as F
from torch.nn import ReLU, Dropout, Linear
from torch_geometric.nn.norm import BatchNorm
import torch_geometric.nn.conv
from torch_geometric.nn.conv import SAGEConv, GraphConv, TAGConv, GATConv, GATv2Conv, ResGatedGraphConv
# Try NNConv, GATv2Conv, GINConv, update dependencies. 
import torch_geometric.nn.dense
import torch_geometric.nn.pool
import tqdm
from progress.bar import Bar
from sklearn.metrics import mean_absolute_error, r2_score
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.data import (Batch, ClusterData, ClusterLoader, Data,
                                  DataLoader, Dataset, GraphSAINTNodeSampler,
                                  GraphSAINTRandomWalkSampler, InMemoryDataset,
                                  NeighborSampler)
from torch_geometric.nn import MessagePassing, Sequential, global_mean_pool
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.utils.convert import from_networkx, to_networkx

import parse

# GLOBALS: 
embed_dim = 128
use_FP16=False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrainNodes:
    def __init__(self,
                 node_id,
                 node_type=None,
                 capacity=None,
                 history_cost=None,
                 src_node=None,
                 sink_node=None,
                 in_netlist=None,
                 initial_cost=None,
                 num_netlists=None,
                 overused=None,
                 startEdge=None,
                 dest_edges=None):
        self.node_id = node_id
        self.history_cost = history_cost
        self.initial_cost = initial_cost
        self.in_netlist = in_netlist
        self.src_node = src_node
        self.num_netlists = num_netlists
        self.overused = overused
        self.sink_node = sink_node
        self.node_type = node_type
        self.capacity = capacity
        if dest_edges is not None:
            if isinstance(dest_edges, str):
                self.dest_edges = ast.literal_eval(dest_edges)
            else:
                self.dest_edges = dest_edges
        elif startEdge is not None:
            self.dest_edges = [startEdge]
        else:
            self.dest_edges = []

    def AddHistory(self, history_cost):
        self.history_cost = history_cost

    def AddNodeType(self, node_type):
        self.node_type = node_type
        
    def AddSrcNode(self, src_node):
        self.src_node = src_node
        
    def AddSinkNode(self, sink_node):
        self.sink_node = sink_node
        
    def AddInNetList(self, in_netlist):
        self.in_netlist = in_netlist
        
    def AddOverused(self, overused):
        self.overused = overused
        
    def AddNumNetlists(self, num_netlists):
        self.num_netlists = num_netlists
        
    def AddCapacity(self, capacity):
        self.capacity = capacity

    def AddPrev(self, initial_cost):
        self.initial_cost = initial_cost

    def AddEdge(self, node_id):
        self.dest_edges.append(node_id)

    def MatchID(self, node_id):
        return self.node_id == node_id

    def ToDict(self):
        return {
            "node_id": self.node_id,
            "dest_edges": self.dest_edges,
            "history_cost": self.history_cost,
            "initial_cost": self.initial_cost,
            "capacity": self.capacity,
            "node_type": self.node_type,
            "in_netlist": self.in_netlist,
            "src_node": self.src_node,
            "sink_node": self.sink_node,

        }

    def GetEdgeIndex(self):
        return [
            int(self.node_id) for i in range(len(self.dest_edges))
        ], \
            [
            int(edge) for edge in self.dest_edges
        ]

    def GetFeatures(self):
        return [float(self.prev_cost)]

    def GetTarget(self):
        return [float(self.history_cost)]


class TrainGraph:
    def __init__(self, bench_name):
        self.bench_name = bench_name
        self.nodes = {}
        self.NodeKeys = ["node_id", "dest_edges", "node_type", "capacity",
                         "initial_Cost", "history_Cost", "src_node",
                         "sink_node", "in_netlist", "num_netlists", "overused"]

    def GetKeys(self):
        return self.NodeKeys

    def GetNodes(self):
        return self.nodes

    def GetBenchName(self):
        return self.bench_name

    def AddNode(self, node_id, history_cost):
        self.nodes[node_id] = TrainNodes(
            node_id, target_history_cost=history_cost)

    def AddEdge(self, src_node, sink_node):
        self.nodes[src_node].AddEdge(sink_node)

    def SafeAddHistoryCost(self, node_id, history_cost):
        # if node_id == '0' or node_id == 0:
        #     print("target_history_cost: ", target_history_cost)
        if node_id in self.nodes:
            self.nodes[node_id].AddHistory(history_cost)
        else:
            self.nodes[node_id] = TrainNodes(
                node_id, history_cost=history_cost)

    def SafeAddCapacity(self, node_id, capacity):
        # if node_id == '0' or node_id == 0:
        #     print("target_history_cost: ", target_history_cost)
        if node_id in self.nodes:
            self.nodes[node_id].AddCapacity(capacity)
        else:
            self.nodes[node_id] = TrainNodes(
                node_id, capacity=capacity)
            
    def SafeAddOverused(self, node_id, overused):
        # if node_id == '0' or node_id == 0:
        #     print("target_history_cost: ", target_history_cost)
        if node_id in self.nodes:
            self.nodes[node_id].AddOverused(overused)
        else:
            self.nodes[node_id] = TrainNodes(
                node_id, overused=overused)
            
    def SafeAddNumNetlists(self, node_id, num_netlists):
        # if node_id == '0' or node_id == 0:
        #     print("target_history_cost: ", target_history_cost)
        if node_id in self.nodes:
            self.nodes[node_id].AddNumNetlists(num_netlists)
        else:
            self.nodes[node_id] = TrainNodes(
                node_id, num_netlists=num_netlists)

    def SafeAddNodeType(self, node_id, node_type):
        # if node_id == '0' or node_id == 0:
        #     print("target_history_cost: ", target_history_cost)
        if node_id in self.nodes:
            self.nodes[node_id].AddNodeType(node_type)
        else:
            self.nodes[node_id] = TrainNodes(
                node_id, node_type=node_type)
            
    def SafeAddSrcNode(self, node_id, src_node):
        # if node_id == '0' or node_id == 0:
        #     print("target_history_cost: ", target_history_cost)
        if node_id in self.nodes:
            self.nodes[node_id].AddSrcNode(src_node)
        else:
            self.nodes[node_id] = TrainNodes(
                node_id, src_node=src_node)
            
    def SafeAddSinkNode(self, node_id, sink_node):
        # if node_id == '0' or node_id == 0:
        #     print("target_history_cost: ", target_history_cost)
        if node_id in self.nodes:
            self.nodes[node_id].AddSinkNode(sink_node)
        else:
            self.nodes[node_id] = TrainNodes(
                node_id, node_type=sink_node)
            
    def SafeAddInNetlist(self, node_id, in_netlist):
        # if node_id == '0' or node_id == 0:
        #     print("target_history_cost: ", target_history_cost)
        if node_id in self.nodes:
            self.nodes[node_id].AddInNetlist(in_netlist)
        else:
            self.nodes[node_id] = TrainNodes(
                node_id, node_type=in_netlist)
            
    def SafeAddInitialCost(self, node_id, initial_cost):
        # if node_id == '0' or node_id == 0:
        #     print("prev_history_cost: ", prev_history_cost)
        if node_id in self.nodes:
            self.nodes[node_id].AddPrev(initial_cost)
        else:
            self.nodes[node_id] = TrainNodes(
                node_id, initial_cost=initial_cost)

    def SafeAddEdge(self, node_id, sink_node):
        # if node_id == '0' or node_id == 0:
        #     print("sink_node: ", sink_node)
        if node_id in self.nodes:
            self.nodes[node_id].AddEdge(sink_node)
        else:
            self.nodes[node_id] = TrainNodes(node_id, startEdge=sink_node)

    def NodeFromDict(self, dict):
        node_id = dict["node_id"]
        self.nodes[node_id] = TrainNodes(
            node_id,
            node_type=dict["node_type"],
            dest_edges=dict["dest_edges"],
            capacity=dict["capacity"],
            initial_cost=dict["initial_cost"],
            history_cost=dict["history_cost"],
            src_node=dict["src_node"],
            sink_node=dict["sink_node"],
            in_netlist=dict['in_netlist'],
            num_netlists=dict['num_netlists'],
            overused=dict['overused']
           )

    def ToDataDict(self):
        # print("DataDict has {} elements".format(len(self.nodes.keys())))
        src_nodes = list()
        dest_nodes = list()
        for node in self.nodes:
            src_index, sink_index = self.nodes[node].GetEdgeIndex()
            src_nodes = src_nodes + src_index
            dest_nodes = dest_nodes + sink_index
        return {
            "x": torch.tensor([[x] for x in itertools.chain.from_iterable(
                self.nodes[node].GetFeatures() for node in self.nodes)],
                dtype=torch.float),
            "y": torch.tensor([[x] for x in itertools.chain.from_iterable(
                self.nodes[node].GetTarget() for node in self.nodes)],
                dtype=torch.float),
            # "edge_index": torch.tensor([x for x in
            # itertools.chain.from_iterable(
            # self.nodes[node].GetEdgeIndex() for node in self.nodes)],
            # dtype=torch.long),
            "edge_index": torch.tensor([[src, dest] for src, dest in zip(
                src_nodes, dest_nodes)],
                dtype=torch.long)
        }


class GNNDataset(Dataset):

    def __init__(self, root, inDir, outDir, transform=None,
                 pre_transform=None, dataExtensions=".csv"):
        # print("Called init")
        self.dataDir = inDir
        self.outDir = outDir
        self.length = len(glob.glob(os.path.join(self.dataDir, "*-nodes*.csv")))
        self.pathdict = glob.glob(os.path.join(self.dataDir, "*-nodes*.csv"))
        self.indices2 = {k: i for i, k in enumerate(self.pathdict)}
        self.dataExtensions = dataExtensions
        super(GNNDataset, self).__init__(root, transform, pre_transform)
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        print("Called raw_file_names")
        return parse.FindSpecificFiles(self.dataDir, self.dataExtensions)

    @property
    def processed_file_names(self):
        # print("Called processed_file_names")
        return ['GNN_Processed_Data_{}.pt'.format(i) for i in
                range(self.length)]

    def download(self):
        print("Called download")
        
    def single_process(self, node_path):
        num = self.indices2[node_path]
        target_path = node_path.partition("-")[0]+"-hcost.csv"
        x, y = parse.parse_node_features(node_path, target_path)
        edge_path = node_path.partition("-")[0]+"-edges.csv"
        edge_index = parse.parse_edge_features(edge_path)
        data = Data(x=x, y=y, edge_index=edge_index)
        torch.save(data, os.path.join(self.processed_paths[num]))

    def process(self):
        print("Called process")
        # pool = Pool(cpu_count())
        paths = glob.glob(os.path.join(self.dataDir, "*-nodes*.csv"))
        with Pool(processes=8) as p:
            with tqdm.tqdm(total=self.length) as pbar:
                for i, _ in enumerate(p.imap_unordered(self.single_process, paths)):
                    pbar.update()
        print("Finished Processing")
        
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(self.processed_paths[idx])
        return data


class GNNVPRL(pl.LightningModule):
    def __init__(self, **kwargs):
        super(GNNVPRL, self).__init__()

        self.num_features = kwargs['num_features'] \
            if "num_features" in kwargs.keys() else 14

        self.num_targets = kwargs['num_features'] \
            if "num_targets" in kwargs.keys() else 1

        self.hidden = 3

        self.GATSequential = Sequential("x, edge_index, batch", [
            (GATv2Conv(self.num_features, self.hidden), "x, edge_index -> x1"),
            (BatchNorm(self.hidden), "x1 -> x1d"),
            (ReLU(), "x1d -> x1d"),
            (GATv2Conv(self.hidden, self.hidden), "x1d, edge_index -> x2"),
            (BatchNorm(self.hidden), "x2 -> x3d"),
            (ReLU(), "x3d -> x3d"),
            (GATv2Conv(self.hidden, self.num_targets), "x3d, edge_index -> x4d")])

        self.TAGSequential = Sequential("x, edge_index, batch", [
            (TAGConv(self.num_features, self.hidden), "x, edge_index -> x1"),
            (BatchNorm(self.hidden), "x1 -> x1t"),
            (ReLU(), "x1t -> x1t"),
            (TAGConv(self.hidden, self.hidden), "x1t, edge_index -> x2"),
            (BatchNorm(self.hidden), "x2 -> x2t"),
            (ReLU(), "x2t -> x2t"),
            (TAGConv(self.hidden, self.num_targets), "x2t, edge_index -> x3t")])
    
        self.hidden2 = 32
        self.SAGESequential = Sequential("x, edge_index, batch", [
            (SAGEConv(self.num_features, self.hidden2), "x, edge_index -> x1"),
            (BatchNorm(self.hidden2), "x1 -> x1s"),
            (ReLU(), "x1s -> x1s"),
            (SAGEConv(self.hidden2, self.hidden2), "x1s, edge_index -> x2"),
            (BatchNorm(self.hidden2), "x2 -> x2s"),
            (ReLU(), "x2s -> x2s"),
            (SAGEConv(self.hidden2, self.num_targets), "x2s, edge_index -> x3s")])

        # self.GraphSequential = Sequential("x, edge_index, batch", [
        #     (GraphConv(self.num_features, self.hidden), "x, edge_index -> x1"),
        #     (BatchNorm(self.hidden), "x1 -> x1g"),
        #     (ReLU(), "x1g -> x1g"),
        #     (GraphConv(self.hidden, self.hidden), "x1g, edge_index -> x2"),
        #     (BatchNorm(self.hidden), "x2 -> x2g"),
        #     (ReLU(), "x2g -> x2g"),
        #     (GraphConv(self.hidden, self.num_targets), "x2g, edge_index -> x3g")])


        # self.Seq = Sequential("x", [
        #     (Linear(self.num_features, self.hidden), "x -> x1"),
        #     (ReLU(), "x1 -> x1r"),
        #     (Linear(self.hidden, self.hidden), "x1r -> x2"),

        # self.NNConv()

        # self.PSequential = Sequential("x, edge_index, batch", [
        


        self.Linear = Linear(3, self.num_targets)

      

        # self.GATSequential.to(device)
        # self.TAGSequential.to(device)
        # self.SAGESequential.to(device)
        # self.GraphSequential.to(device)
        self.total_nodes = 0

       

    def forward(self, data):
        # data = data.to(device)
        x, edge_index = data.x, data.edge_index
        x1 = self.GATSequential(x, edge_index, data.batch)
        x2 = self.TAGSequential(x, edge_index, data.batch)
        x3 = self.SAGESequential(x, edge_index, data.batch)
        # x4 = self.GraphSequential(x, edge_index, data.batch)
        x4 = torch.cat((x1, x2, x3), dim=1)
        x = global_mean_pool(x, data.batch)
        x = self.Linear(x4)
        x = F.relu(x)
        x_i = F.dropout(x, p=0.95)
        x = torch.where(data.y == 0, x_i, x)
       
        return x

    def predict_step(self, data, data_idx, dataloader_idx):
        return self(data)

    def training_step(self, data, batch_idx):
        y_pred = self.forward(data)
        loss = torch.nn.SmoothL1Loss()(y_pred, data.y)
        return {'loss': loss}

    def validation_step(self, data, batch_idx):
        y_pred = self.forward(data)
        loss = torch.nn.SmoothL1Loss()(y_pred, data.y)
        # num_nodes = data.num_nodes
        return {'val_loss': loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=3e-3, 
            weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,  
            patience=3, 
            verbose=True)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'frequency': 1
            }
           }


    def test_step(self, data, batch_idx):
        y_pred = self.forward(data)
        loss = torch.nn.SmoothL1Loss()(y_pred, data.y)
        # return y_pred
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        self.log("Test Loss: {}".format(outputs['test_loss']))
class GNNVPR(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNVPR, self).__init__()
     
      
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hidden_channels))
        self.convs.append(GATv2Conv(hidden_channels, hidden_channels))
        self.convs.append(GATv2Conv(hidden_channels, hidden_channels))
        self.convs.append(GATv2Conv(hidden_channels, out_channels))
        self.num_layers = len(self.convs)

        self.convs2 = torch.nn.ModuleList()
        self.convs2.append(TAGConv(in_channels, hidden_channels))
        self.convs2.append(TAGConv(hidden_channels, hidden_channels))
        self.convs2.append(TAGConv(hidden_channels, out_channels))
        self.num_layers2 = len(self.convs2)

        self.convs3 = torch.nn.ModuleList()
        self.convs3.append(SAGEConv(in_channels, hidden_channels))
        self.convs3.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs3.append(SAGEConv(hidden_channels, out_channels))
        self.num_layers3 = len(self.convs3)


        

        # self.convs4 = torch.nn.ModuleList()
        # self.convs4.append(ResGatedGraphConv(in_channels, hidden_channels))
        # self.convs4.append(ResGatedGraphConv(hidden_channels, hidden_channels))
        # self.convs4.append(ResGatedGraphConv(hidden_channels, out_channels))
        # self.num_layers4 = len(self.convs)


        self.linear = torch.nn.Linear(3, 1)

    def forward(self, data):
        # print(dir(data))
        x, edge_index = data.x, data.edge_index
        x_1 = torch.clone(x)
        
        # x_4 = torch.clone(x)
        for i in range(self.num_layers):
            x_1 = self.convs[i](x_1, edge_index)
            if i != self.num_layers - 1:
                x_1 = F.relu(x_1)
        
        x_2 = torch.clone(x)
        for i in range(self.num_layers2):
            x_2 = self.convs2[i](x_2, edge_index)
            if i != self.num_layers2 - 1:
                x_2 = F.relu(x_2)


        x_3 = torch.clone(x)
        for i in range(self.num_layers3):
            x_3 = self.convs3[i](x_3, edge_index)
            if i!= self.num_layers3 - 1:
                x_3 = F.relu(x_3)


        x = torch.cat([x_1, x_2, x_3], dim=1)
         

        # x = x_1
        x = self.linear(x)
        x = F.relu(x)
        x_i = F.dropout(x, p=0.95)
        x = torch.where(data.y == 0., x_i, x)
        return x

        # alu4, des, clma, s38417
def main(options):
    
    def train():
        model.train()
        loss_all = 0
        total_nodes = 0
        # for loader in train_loader:
        #     # loader = GraphSAINTNodeSampler(loader, batch_size=6000, num_steps=5)
        #     loader = 

        ## Fix the Optimizer here, lol. 
        for data in train_loader:
            # data = data.to(device)
            output = model(data)
            # target = data.y.to(device)
            loss = torch.nn.SmoothL1Loss()(output.to(torch.float32), target)
            # loss.backward()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_all += loss.item() * data.num_nodes
            total_nodes += data.num_nodes
            # scalar.update()
        
        return loss_all / total_nodes

    

    def evaluate(loader):
        model.eval()
        maes = []
        with torch.no_grad():
            # for loader in train_loader:
            #     loader = GraphSAINTNodeSampler(loader, batch_size=6000, num_steps=5)
            for load in train_loader:
                # load = load.to(device)
                pred = model(load).detach().cpu().numpy()
                target = load.y.detach().cpu().numpy()
                maes.append(mean_absolute_error(target, pred))               
        return sum(maes) / len(maes)

    print("Initializing Dataset & Batching")
    dataset = GNNDataset(options.inputDirectory,
                         options.inputDirectory, options.outputDirectory)
    dataset = dataset.shuffle()
    one_tenth_length = int(len(dataset) * 0.1)
    train_dataset = dataset[:one_tenth_length * 8]
    val_dataset = dataset[one_tenth_length * 8:one_tenth_length * 9]
    test_dataset = dataset[one_tenth_length * 9:]
    len(train_dataset), len(val_dataset), len(test_dataset)
    print("Done")

    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=6)
    # val_loader = val_dataset
    # test_loader = test_dataset

    print("Starting Training: on device: ", device)
    lightning_model = GNNVPRL()

    num_epochs = 10
    val_check_interval = len(train_loader)

    trainer = pl.Trainer(accelerator='gpu',
                         precision=16,
                         max_epochs=num_epochs,
                        #  gradient_clip_val=1,
                         val_check_interval=val_check_interval,
                         devices=1,
                         callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)])
                         
    # trainer.tune(lightning_model, train_loader, val_loader)                
    trainer.fit(lightning_model, train_loader, val_loader)
    trainer.save_checkpoint('model.ckpt', weights_only=True)



    # model = GNNVPR(14, 3, 1).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
    #                              weight_decay=5e-4)
    # scalar = GradScaler()
    # initial = time.perf_counter()
    # loss = train()
    # for epoch in range(1, 10000):
    #     loss = train()
    #     train_loss = evaluate(train_loader)
    #     val_loss = evaluate(val_loader)
    #     test_loss = evaluate(test_loader)
    #     run = time.perf_counter() - initial
    #     print(('Epoch: {:03d}, Loss: {:.5f}, Train MAE: {:.5f},' +
    #             'Val MAE: {:.5f}, Test MAE: {:.5f},' +
    #             'Time: {:.2f}').format(epoch, loss,
    #                                     train_loss,
    #                                     val_loss,
    #                                     test_loss,
    #                                     run))
    #     torch.save(model.state_dict(), "model.pt")

    return
    
if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-I", "--inputDirectory", dest="inputDirectory",
                      help="directory that contains the benchmarks to be run",
                      metavar="INPUT")
    parser.add_option("-O", "--outputDirectory", dest="outputDirectory",
                      help="directory to output the " +
                      "completed model and metrics",
                      metavar="OUTPUT")
    parser.add_option("-r", "--rootDirectory", dest="rootDirectory",
                      help="directory to output the " +
                      "completed model and metrics",
                      metavar="OUTPUT")
    (options, args) = parser.parse_args()
    # calling main function
    main(options)

