"""Trains a GNN using PyTorchGeometric given a parsed graph dataset.

Returns:
    pkl file: an output model file as well as training results.
"""
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
import torch.nn.functional as F
import torch_geometric.nn.conv
from torch_geometric.nn.conv import SAGEConv, GraphConv, TAGConv
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
from torch_geometric.nn import MessagePassing
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.utils.convert import from_networkx, to_networkx
from tqdm import *

import parse

embed_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

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
        # print("Called raw_file_names")
        # return parse.FindSpecificFiles(self.dataDir)
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
            with tqdm(total=self.length) as pbar:
                for i, _ in enumerate(p.imap_unordered(self.single_process, paths)):
                    pbar.update()
        print("Finished Processing")
        
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(self.processed_paths[idx])
        # data = ClusterData(data, num_parts=128)
        # data_loader = ClusterLoader(data)
        data_loader = GraphSAINTNodeSampler(data, batch_size=6000, num_steps=5)
        return data_loader


class GraNNy_ViPeR(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraNNy_ViPeR, self).__init__()
     
        self.num_layers = 3
        self.convs = torch.nn.ModuleList()
        self.convs.append(TAGConv(in_channels, hidden_channels))
        self.convs.append(TAGConv(hidden_channels, hidden_channels))
        self.convs.append(TAGConv(hidden_channels, out_channels))
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, data):
        # print(dir(data))
        x, edge_index = data.x, data.edge_index
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = torch.sigmoid(x)
        # # Label Normalization
        x = self.linear(x)
        x =  F.relu(x)
        indices = torch.randperm(len(x))[:int(len(x)*.98)]
        zeros = torch.zeros(len(x[indices]), 1).to(device)
        x[indices] = torch.where(data.y[indices] == 0., zeros, x[indices])
        return x


    def inference(self, data_loader):
        # Here we are sent the entire subgraph loader, and compute per that
        # TODO: Refactor.
        # x, edge_index = data.x, data.edge_index
        for i in range(self.num_layers):
            xs = []
            for data in data_loader:
                data = data.to(device)
                x, edge_index = data.x, data.edge_index
                x = self.convs[i](x, edge_index)
                if i != self.num_layers - 1:
                    x = F.sigmoid(x)
                else:
                    x = self.linear(x)
                    x =  F.relu(x)
                    indices = torch.randperm(len(x))[:int(len(x)*.98)]
                    zeros = torch.zeros(len(x[indices]), 1).to(device)
                    x[indices] = torch.where(data.y[indices] == 0., zeros, x[indices])
                    xs.append(x.cpu())
        # # Label Normalization
            x_all = torch.cat(xs, dim=0)
        return x_all
        


use_FP16=False

def main(options):
    def train():
        model.train()
        loss_all = 0
        total_nodes = 0
        for loader in train_loader:
            loss_local = 0
            nodes_local = 0
            for data in loader:
                data = data.to(device)
                optimizer.zero_grad()
                output = model(data)
                target = data.y.to(device)
                loss = torch.nn.SmoothL1Loss()(output.to(torch.float32), target)
                scalar.scale(loss).backward()
                scalar.step(optimizer)
                loss_local += loss.item() * data.num_nodes
                nodes_local += data.num_nodes
                scalar.update()
            total_nodes += nodes_local
            loss_all +=  loss_local
        return loss_all / total_nodes

    def evaluate(loader):
        model.eval()
        maes = []
        with torch.no_grad():
            for loader in train_loader:
                for load in loader:
                    load = load.to(device)
                    pred = model(load).detach().cpu().numpy()
                    target = data.y.detach().cpu().numpy()
                    maes.append(mean_absolute_error(target, pred))               
        # predictions = np.hstack(predictions)
        # targets = np.hstack(targets)
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

    train_loader = train_dataset
    val_loader = val_dataset
    test_loader = test_dataset


    print("Starting Training: on device: ", device)
    model = GraNNy_ViPeR(14, 128, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,
                                 weight_decay=5e-4)
    scalar = GradScaler()
    initial = time.perf_counter()
    for epoch in range(1, 200):
        loss = train()
        train_loss = evaluate(train_loader)
        # train_loss = -1
        val_loss = evaluate(val_loader)
        # val_loss = -1
        
        
        test_loss = evaluate(test_loader)
        # test_loss = -1
        run = time.perf_counter() - initial
        if (epoch % 10 == 0) or epoch == 1:
            print(('Epoch: {:03d}, Loss: {:.5f}, Train MAE: {:.5f},' +
                   'Val MAE: {:.5f}, Test MAE: {:.5f},' +
                   'Time: {:.2f}').format(epoch, loss,
                                          train_loss,
                                          val_loss,
                                          test_loss,
                                          run))
            torch.save(model.state_dict(), "model.pt")

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

