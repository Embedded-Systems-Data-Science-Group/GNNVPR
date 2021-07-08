"""Runs Inference on a specified input benchmark from a specified model.
"""
import PyTorchGeometricTrain
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
from torch_geometric.data import DataLoader

embed_dim = 128


# from PyTorchGeometricTrain import GraNNy_ViPeR
# from PyTorchGeometricTrain import GNNDataset

def main(options):
    device = "cpu"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    t1 = time.time()
    proc_dir = options.inputDirectory+"processed/"
    for file in glob.glob(os.path.join(proc_dir, "*.pt")):
        os.remove(file)
    dest_dir = options.inputDirectory+"raw/"
    for file in glob.glob(os.path.join(options.inputDirectory, "*.csv")):
        shutil.copy(file, os.path.join(dest_dir, os.path.basename(file)))
    print("--- Loading CSV Data took %s seconds ---" % (time.time() - t1))
    t2 = time.time()
    model = PyTorchGeometricTrain.GraNNy_ViPeR().to(device)
    model.load_state_dict(torch.load('/home/spicygremlin/Github/CS220/model.pt'))
    model.eval()
    print("--- Model Loading took %s seconds ---" % (time.time() - t2))
    t4 = time.time()
    dataset = PyTorchGeometricTrain.GNNDataset(options.inputDirectory,
                                               options.inputDirectory,
                                               options.outputDirectory)
    print("--- Processing CSV took %s seconds ---" % (time.time() - t4))
    t3 = time.time()
    test_loader = DataLoader(dataset, batch_size=1)
    for data in test_loader:
        data = data.to(device)
        pred = model(data).detach().cpu().numpy()
        # print(pred)
        df = pd.DataFrame.from_dict(pred)
        # print("honkers")
        # df = (df-df.min())/(df.max() - df.min())
       
        # def sigmoid(x):
        #     return 1 / (1.0 + math.exp(-x))
        # df = df.apply(sigmoid, axis=1)
        # df = ((df+0.5) ** 4)
        # df = (df * 15) ** 2
        # df = df * 20
        df = df * 4
        df = df + 1
        # df = df + 1
        # print(df.head())
        # df = df.apply(np.floor)
        # df = df * 4
        # print(df)
        # print("Saving file to: ", os.getcwd())
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
