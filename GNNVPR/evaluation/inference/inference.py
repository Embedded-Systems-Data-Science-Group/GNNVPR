"""Runs Inference on a specified input benchmark from a specified model.
"""

import PyTorchGeometricTrain
from optparse import OptionParser
import pandas as pd
import torch
from torch_geometric.data import DataLoader

embed_dim = 128


# from PyTorchGeometricTrain import GraNNy_ViPeR
# from PyTorchGeometricTrain import GNNDataset

def main(options):
    model = PyTorchGeometricTrain.GraNNy_ViPeR().to('cuda')
    model.load_state_dict(torch.load('model.pt'))
    model.eval()

    dataset = PyTorchGeometricTrain.GNNDataset(options.inputDirectory,
                                               options.inputDirectory,
                                               options.outputDirectory)
    test_loader = DataLoader(dataset, batch_size=1)

    for data in test_loader:
        data = data.to('cuda')
        pred = model(data).detach().cpu().numpy()
        print(pred)
        df = pd.DataFrame.from_dict(pred)
        df.to_csv('output.csv', index=False)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-i", "--inputDirectory", dest="inputDirectory",
                      help="directory that contains the benchmarks to be run",
                      metavar="INPUT")
    # ! Add an option to load in a saved model. 
    parser.add_option("-o", "--outputDirectory", dest="outputDirectory",
                      help="directory to output the completed model" +
                      "and metrics",
                      metavar="OUTPUT")
    parser.add_option("-r", "--rootDirectory", dest="rootDirectory",
                      help="directory to output the completed model" +
                      "and metrics",
                      metavar="OUTPUT")
    (options, args) = parser.parse_args()
    # calling main function
    main(options)
