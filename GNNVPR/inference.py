"""Runs Inference on a specified input benchmark from a specified model.
"""
from optparse import OptionParser
import pandas as pd
import torch
import glob
import time
import os
import numpy as np
import math
import shutil
from shutil import copyfile
import ast
import glob
import itertools
import os
import time
from multiprocessing import Pool, cpu_count, freeze_support
from optparse import OptionParser
import model
import networkx as nx
import torch
# import 
import torch.nn.functional as F
import torch_geometric.nn.conv
from torch_geometric.nn.conv import SAGEConv, GraphConv, TAGConv, GATConv
import torch_geometric.nn.dense
import torch_geometric.nn.pool
import tqdm
from progress.bar import Bar
import pytorch_lightning as pl
from pytorch_lightning import Trainer
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
from torch_geometric.data import DataLoader

embed_dim = 128


# from PyTorchGeometricTrain import GraNNy_ViPeR
# from PyTorchGeometricTrain import GNNDataset

def main(options):
    # device = "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    t1 = time.time()
    proc_dir = options.inputDirectory+"processed/"
    for file in glob.glob(os.path.join(proc_dir, "*.pt")):
        os.remove(file)
    dest_dir = options.inputDirectory+"raw/"
    for file in glob.glob(os.path.join(options.inputDirectory, "*.csv")):
        shutil.copy(file, os.path.join(dest_dir, os.path.basename(file)))
    print("--- Loading CSV Data took %s seconds ---" % (time.time() - t1))
    t2 = time.time()
    my_model = model.GNNVPRL()
    trainer = Trainer(accelerator="gpu",
                      precision=16,
                      devices=1)
    # model2 = model.GNNVPRL().to(device)
    model2 = my_model.load_from_checkpoint('/mnt/e/benchmarks/model-perfect.ckpt')
    # model2 = model2.to(device)
    model2.eval()
    # model2.load_state_dict(torch.load('/mnt/e/benchmarks/model.ckpt'))
    # model2.eval()

    # for param in model2.parameters():
    #     param.grad = None
    # print("--- Model Loading took %s seconds ---" % (time.time() - t2))
    t4 = time.time()
    dataset = model.GNNDataset(options.inputDirectory,
                                               options.inputDirectory,
                                               options.outputDirectory)
    # print("--- Processing CSV took %s seconds ---" % (time.time() - t4))
    t3 = time.time()
    # test_loader = DataLoader(dataset, batch_size=1)
    test_loader = dataset
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
   
    for data in test_loader:
        # for data in loader:
        # data = data.to(device)
        with torch.no_grad():
            pred = model2(data).detach().cpu().numpy()
    
        t5 = time.time()
        # print("--- Direct Model Inference took %s seconds ---" % (time.time() - t5))
        df = pd.DataFrame.from_dict(pred)
        df = df + 1
        
        df.to_csv('prediction.csv', index=False,
                    header=False)
    print("--- Prediction & Saving took %s seconds ---" % (time.time() - t3))

if __name__ == "__main__":
    t4 = time.time()
    parser = OptionParser()
    parser.add_option("-i", "--inputDirectory", dest="inputDirectory",
                      help="directory that contains the benchmarks to be run",
                      metavar="INPUT")
    # ! Add an option to load in a saved model. 
    parser.add_option("-o", "--outputDirectory", dest="outputDirectory",
                      help="directory to output the completed model" +
                      "and metrics",
                      metavar="OUTPUT")
    # Get Arch Name
    #
    # Get Circuit Name
    #
    parser.add_option("-r", "--rootDirectory", dest="rootDirectory",
                      help="directory to output the completed model" +
                      "and metrics",
                      metavar="OUTPUT")
    (options, args) = parser.parse_args()
    # calling main function
    main(options)
    print("--- Total Script Time: %s seconds ---" % (time.time() - t4))
