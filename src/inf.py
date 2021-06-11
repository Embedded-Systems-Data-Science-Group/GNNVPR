import os

import torch, csv, itertools, ast
import torch_geometric.nn.conv
import pandas as pd
import parse_xml_rr_graph_to_csv
import torch.nn.functional as F
import numpy as np
from optparse import OptionParser
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import max_error


from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn import MessagePassing, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import remove_self_loops, add_self_loops

embed_dim = 128
import PyTorchGeometricTrain
# from PyTorchGeometricTrain import GraNNy_ViPeR
# from PyTorchGeometricTrain import GNNDataset

def main(options):
    model = PyTorchGeometricTrain.GraNNy_ViPeR().to('cuda')
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    
    dataset = PyTorchGeometricTrain.GNNDataset(options.inputDirectory,  options.inputDirectory, options.outputDirectory)
    test_loader = DataLoader(dataset, batch_size=1)

    for data in test_loader:
        data = data.to('cuda')
        pred = model(data).detach().cpu().numpy()
        print(pred)
        df = pd.DataFrame.from_dict(pred)
        df.to_csv('output.csv', index=False)
    

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-I", "--inputDirectory", dest="inputDirectory",
                      help="directory that contains the benchmarks to be run", metavar="INPUT")
    parser.add_option("-O", "--outputDirectory", dest="outputDirectory",
                      help="directory to output the completed model and metrics", metavar="OUTPUT")
    parser.add_option("-r", "--rootDirectory", dest="rootDirectory",
                      help="directory to output the completed model and metrics", metavar="OUTPUT")
    (options, args) = parser.parse_args()
    # calling main function
    main(options)