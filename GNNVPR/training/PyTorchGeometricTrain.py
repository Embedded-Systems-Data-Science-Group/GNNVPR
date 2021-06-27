"""Trains a GNN using PyTorchGeometric given a parsed graph dataset.

Returns:
    pkl file: an output model file as well as training results.
"""
import ast
import itertools
from optparse import OptionParser
import os
import glob
import torch
import gc
import torch.nn.functional as F
import torch_geometric.nn.conv
from sklearn.metrics import mean_absolute_error
from torch_geometric.data import Data, DataLoader, InMemoryDataset
from torch_geometric.data import Data, DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops
import parse

embed_dim = 128


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
        self.length = len(glob.glob(os.path.join(self.dataDir, "*-nodes.csv")))
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

    def process(self):
        print("Called process")
        data_list = list()
        num = 0
        for node_path in glob.iglob(os.path.join(self.dataDir, "*-nodes.csv")):
           
            # print("processing... ", raw_path)
            # graph = parse.parse_one_first_last_csv(raw_path)
            # inputDict = graph.ToDataDict()
            x, y = parse.parse_node_features(node_path)
            
            edge_path = node_path.partition("-")[0]+"-edges.csv"
            
            edge_index = parse.parse_edge_features(edge_path)
            data = Data(x=x, y=y, edge_index=edge_index)
            data_list.append(data)

            # data, slices = self.collate(data_list)
            # print(self.processed_paths[num])
            torch.save(data, os.path.join(self.processed_paths[num]))
            num += 1
            del data
            del x
            del y
            del edge_index
            gc.collect()
        # print(self.raw_paths)
        
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(self.processed_paths[idx])
        return data


class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        # "Max" aggregation.
        super(SAGEConv, self).__init__(aggr='max', node_dim=-1)
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.act = torch.nn.ReLU()
        self.update_lin = torch.nn.Linear(
            in_channels + out_channels, in_channels, bias=False)
        self.update_act = torch.nn.ReLU()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        
        # x_j has shape [E, in_channels]

        x_j = self.lin(x_j)
        x_j = self.act(x_j)

        return x_j

    def update(self, aggr_out, x):
        
        # aggr_out has shape [N, out_channels]

        new_embedding = torch.cat([aggr_out, x], dim=1)

        new_embedding = self.update_lin(new_embedding)
        new_embedding = self.update_act(new_embedding)

        return new_embedding


class GraNNy_ViPeR(torch.nn.Module):
    def __init__(self):
        super(GraNNy_ViPeR, self).__init__()
        self.NUM_FEATURES = 11
        # self.conv1 = SAGEConv(1, 128)
        self.conv1 = torch_geometric.nn.conv.SAGEConv(self.NUM_FEATURES, 128)
        self.conv2 = torch_geometric.nn.conv.SAGEConv(128, 128)
        self.conv3 = torch_geometric.nn.conv.SAGEConv(128, 1)
        self.Tconv1a = torch_geometric.nn.conv.TAGConv(self.NUM_FEATURES, 8)
        self.Tconv2a = torch_geometric.nn.conv.TAGConv(8, 1)
        
        self.Tconv1b = torch_geometric.nn.conv.TAGConv(self.NUM_FEATURES, 8,
                                                       K=6)
        self.Tconv2b = torch_geometric.nn.conv.TAGConv(8, 1, K=6)
        # self.pool1 = TopKPooling(128, ratio=0.8)
        # self.pool2 = TopKPooling(128, ratio=0.8)
        # self.pool3 = TopKPooling(128, ratio=0.8)
        # self.item_embedding = torch.nn.Embedding(num_embeddings=7,
        # embedding_dim=embed_dim)
        self.lin1 = torch.nn.Linear(3, 1)
        # self.lin2 = torch.nn.Linear(2, 1)
        self.sig1a = torch.nn.Sigmoid()
        self.sig1b = torch.nn.Sigmoid()

        self.sig2 = torch.nn.Sigmoid()
        
        self.sig3 = torch.nn.Sigmoid()
        self.sig4 = torch.nn.Sigmoid()
        # self.lin1 = torch.nn.Linear(256, 128)
        # self.lin2 = torch.nn.Linear(128, 64)
        # self.lin3 = torch.nn.Linear(64, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index,  = data.x, data.edge_index
        # x = x.squeeze(1)
        # batch = data.batch
        # * Layer 1
        x1 = self.conv1(x, edge_index)
        # Sigmoid Here
        x1 = self.sig1a(x1)
        x1 = self.conv1(x, edge_index)
        # Sigmoid Here
        x1 = self.sig1a(x1)
        x1 = self.conv1(x, edge_index)
        # Sigmoid Here
        x1 = self.sig1a(x1)
        
        x1 = self.conv2(x1, edge_index)
        # Sigmoid Here
        x1 = self.sig1b(x1)
        x1 = F.relu(self.conv3(x1, edge_index))
        
        # * Layer 2
        x2 = self.Tconv1a(x, edge_index)
        # Sigmoid Here
        x2 = self.sig2(x2)
        x2 = F.relu((self.Tconv2a(x2, edge_index)))

        # * Layer 3
        x3 = self.Tconv1b(x, edge_index)
        # Sigmoid here
        x3 = self.sig3(x3)
        x3 = F.relu((self.Tconv2b(x3, edge_index)))
        
        # x2 = torch.cat((x2, x3), dim=1)
        # x2 = F.relu(self.lin2(x2))
        
        x = torch.cat((x1, x2, x3), dim=1)
        x = F.relu(self.lin1(x))
        # x = torch.exp(x)
        # x = (0.5 * x) ** 4
        # x = self.sig4(self.lin1(x))
        # x = torch.nn.functional.normalize(x)
        # x = x/x.sum(0).expand_as(x) 
        # x = x*4

        # x = F.dropout(x, p=0.5, training=self.training)
        return x


def main(options):

    def train():
        model.train()

        loss_all = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            with autocast(enabled=True):
                
                output = model(data)

                target = data.y.to(device)
            # loss = torch.nn.BCEWithLogitsLoss()(output.to(torch.float32),
            # target)
                loss = torch.nn.MSELoss()(output.to(torch.float32), target)
            # loss = torch.nn.MSELoss(reduce=True)(output.to(torch.float32),
            # target)
            if True:
                scalar.scale(loss).backward()
                scalar.step(optimizer)
                loss_all += data.num_graphs * loss.item()
                scalar.update()
            else:
                loss.backward()
                loss_all += data.num_graphs * loss.item()
                optimizer.step()
                                  
        return loss_all / len(train_dataset)

    def evaluate(loader):
        model.eval()

        maes = []

        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                with autocast(enabled=True):
                    pred = model(data).detach().cpu().numpy()

                    target = data.y.detach().cpu().numpy()
                # predictions.append(pred)
                # targets.append(target)
                    maes.append(mean_absolute_error(target, pred))               

        # predictions = np.hstack(predictions)
        # targets = np.hstack(targets)
    
        return sum(maes) / len(maes)

    dataset = GNNDataset(options.inputDirectory,
                         options.inputDirectory, options.outputDirectory)

    dataset = dataset.shuffle()
    one_tenth_length = int(len(dataset) * 0.1)
    train_dataset = dataset[:one_tenth_length * 8]
    val_dataset = dataset[one_tenth_length * 8:one_tenth_length * 9]
    test_dataset = dataset[one_tenth_length * 9:]
    len(train_dataset), len(val_dataset), len(test_dataset)

    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # num_items = df.item_id.max() + 1
    # num_categories = df.category.max() + 1
    # num_items, num_categories

    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraNNy_ViPeR().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,
                                 weight_decay=5e-4)
    scalar = GradScaler()
    for epoch in range(1, 200):
        loss = train()
        train_loss = evaluate(train_loader)
        
        val_loss = evaluate(val_loader)
        
        test_loss = evaluate(test_loader)
        
        if (epoch % 10 == 0) or epoch == 1:
            print(('Epoch: {:03d}, Loss: {:.5f}, Train MAE: {:.5f},' +
                  'Val MAE: {:.5f}, Test MAE: {:.5f}').format(epoch, loss,
                                                              train_loss,
                                                              val_loss,
                                                              test_loss))
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
