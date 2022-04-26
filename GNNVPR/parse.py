"""Takes Collected Data & ensures format is good for DataLoader

Returns:
    csv: A combined csv containining graph features.
"""
import ast
import csv
import os
import pandas as pd
import time
import re
from sklearn.preprocessing import OneHotEncoder
from functools import lru_cache
import networkx as nx
# import matplotlib.pyplot as plt
from optparse import OptionParser

import torch
import numpy as np
from progress.bar import Bar

BENCH_NAME_STRING = r"\\([0-9A-Za-z]+)_[0-9A-Za-z]+.xml"
FIRST_LAST_PARSE_STRING = r'''([0-9A-Za-z]+)_
                                (?:(first)_([0-9A-Za-z\.]+)|([0-9A-Za-z\.]+))_
                                ([0-9A-Za-z]+).csv'''
CSV_FILE_STRING = r"([0-9A-Za-z]+).csv"
CSV_FILE_STRING_NODE = r"([0-9A-Za-z]+)_nodes.csv"
CSV_FILE_STRING_EDGES = r"([0-9A-Za-z]+)_edges.csv"

# Takes in a data class, converts to networkx, adds feature, converts back

def FindSpecificFiles(directory, extension):
    return [f for f in os.listdir(directory) if f.endswith(extension)]


def parse_edge_features(f):
    with open(f) as cF:
        start_time = time.time()
        df = pd.read_csv(f)
        edge_index = [[], []]
        start_time = time.time()   
        edge_index[0] = df['src_node'].values
        edge_index[1] = df['sink_node'].values

    return torch.tensor(edge_index, dtype=torch.long)

 
def parse_node_features(f, g):
    with open(f) as cF:
        df = pd.read_csv(cF)
        df = df.drop(['node_id'], axis=1)
        one_hot = pd.get_dummies(df['node_type'])
        df = df.drop(['node_type'], axis=1)
        df = df.join(one_hot)
        df = df.apply(pd.to_numeric)
        x = df.values
    # Target
    with open(g) as cG:
        df = pd.read_csv(cG)
        # *  We want default values to appear as 0 to the NN.
        df['present_cost'] = df['present_cost'] - 1
        y = df['present_cost'].values.tolist()
        y = [[i] for i in y]
    return torch.tensor(x, dtype=torch.float),\
        torch.tensor(y, dtype=torch.float)


def main(options):
    print("To be Implemented.")
   


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-i", "--inputDirectory", dest="inputDirectory",
                      help="directory that contains the benchmarks to be run",
                      metavar="INPUT")
    parser.add_option("-h", "--historyCostDirectory",
                      dest="historyCostDirectory",
                      help="directory that contains the historyCosts to " /
                            "train for",
                      metavar="HISTORY")
    parser.add_option("-o", "--outputDirectory", dest="outputDirectory",
                      help="directory that contains the combined data files" /
                            "for training",
                      metavar="OUTPUT")
    (options, args) = parser.parse_args()
    # calling main function
    main(options)
