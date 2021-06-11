
import pandas as pd
import argparse
import collections
import time
import numpy as np
import networkx as nx
import os
import csv
# import cupy as np
import torch as th
import torch.nn.functional as F
import torch.nn.init as INIT
import torch.optim as optim
from torch.utils.data import DataLoader

import dgl
from dgl.data.tree import SSTDataset

# from tree_lstm import TreeLSTM

def main():

    edges_data = "datasets/earch_first_alu4_edgelist.csv"
    nodes_data_last_f ="datasets/earch_first_alu4_historycosts.csv"
    nodes_data_first_f ="datasets/earch_last_alu4_historycosts.csv"
    # print(edges_data.head())
    # G = nx.graph()
    
    df1 = pd.read_csv(nodes_data_first_f,index_col=0)
    df1.index = df1.index + 1
    print(df1.head())
    
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
    gr = dgl.from_networkx(G,['First Cost','Last Cost'],device='cuda:0')
    # str2 = "earch_first_alu4_historycosts.csv"
    # str3 = "earch_last_alu4_historycosts.csv"
    
    
    
if __name__=="__main__":
    main()
    
    