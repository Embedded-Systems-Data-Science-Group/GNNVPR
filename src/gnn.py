import dgl
import dgl.data
from dgl.nn import GraphConv
import torch
import networkx as nx
import torch.nn as nn
import torch_scatter.
import torch.nn.functional as F
import time
from pympler import asizeof
import pandas as pd
import csv

import sys
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
    
def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['First Cost'].double()
    labels = g.ndata['Last Cost'].double()
    train_mask_idx = range(0,25000)
    val_mask_idx  = range(25000,30000)
    test_mask_idx = range(30000,40676)
    
    train_mask = torch.DoubleTensor(train_mask_idx)
    val_mask = torch.DoubleTensor(val_mask_idx)
    test_mask = torch.DoubleTensor(test_mask_idx)

    for e in range(100):
        # Forward
        logits = model(g, features).double()
        print(len)
        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        # Accuracy of Training vs test (NOT LABELS)
        train_acc = (pred[train_mask] == labels[train_mask]).double().mean()
        # Validation Se
        val_acc = (pred[val_mask] == labels[val_mask]).double().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).double().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if e % 5 == 0:
        #     print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
        #         e, loss, val_acc, best_val_acc, test_acc, best_test_acc))   
def testDGL_CPU():
    dataset = dgl.data.CoraGraphDataset()
    g = dataset[0]
    model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes)
    train(g, model)

def testDGL_GPU():
    dataset = dgl.data.CoraGraphDataset()
    g = dataset[0]
    g = g.to('cuda')
    model = GCN(g.ndata['feat'].shape[1].float(), 16, dataset.num_classes).to('cuda')
    train(g, model)
    
def yolo():
    edges_data = "datasets/earch_first_alu4_edgelist.csv"
    nodes_data_last_f ="datasets/earch_first_alu4_historycosts.csv"
    nodes_data_first_f ="datasets/earch_last_alu4_historycosts.csv"
    # print(edges_data.head())
    # G = nx.graph()
    
    df1 = pd.read_csv(nodes_data_first_f,index_col=0)
    df1.index = df1.index + 1
    
    df2 = pd.read_csv(nodes_data_last_f,index_col=0)
    df2.index = df2.index + 1

    node_df = pd.concat([df1,df2],axis =1)    
    node_df.columns = ["First Cost", "Last Cost"]
        
    with open(edges_data, 'r') as f:
        edge_data = [(int(line['src_node']),int(line['sink_node'])) for line in  csv.DictReader(f)]
    G = nx.from_edgelist(edge_data)
    
    # print(node_df['First Cost'][1])    
  
    for i in sorted(G.nodes()):
        G.nodes[i]['First Cost'] = node_df['First Cost'].iloc[i]
        G.nodes[i]['Last Cost'] = node_df['Last Cost'].iloc[i]
    g = dgl.from_networkx(G,['First Cost','Last Cost'])
    

    model = GCN(40676,40676,1)
    train(g, model)
    
def main():
    # testDGL()
    # print("Running CPU Cora Dataset")
    # start_time = time.time()
    # testDGL_CPU()
    # print("CPU Model Took %.2f seconds" % (time.time() - start_time))
    
    # print("Running GPU Cora Dataset")
    # start_time = time.time()
    # testDGL_GPU()
    # print("GPU Model Took %.2f seconds" % (time.time() - start_time))
    print("Running Yolo Dataset")
    start_time = time.time()
    yolo()
    print("GPU Model Took %.2f seconds" % (time.time() - start_time))
if __name__ == "__main__":
  
    # calling main function
    main()