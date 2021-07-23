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

import PyTorchGeometricTrain
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

CACHE_CUTOFF = 28000
# Takes in a data class, converts to networkx, adds feature, converts back


def add_khops_feature(graph):
    #  We want to add a feature to each node that represents the distance
    #  to the nearest node that is in the netlist.
    # print(graph[1])
    print(list(graph.nodes(data=True)))
    return graph


def parse_first_last_files(directory, outputDirectory):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    directory : string
        path to a directory of CSV files.
    outputDirectory : string
        [description]
    """    
    fileList = [os.path.join(directory, fileDir)
                for fileDir in os.listdir(directory)]
    matchList = [re.search(FIRST_LAST_PARSE_STRING, filename)
                 for filename in os.listdir(directory)]

    graphs = dict()

    # for match, filename in Bar('Parsing...').iter(zip(matchList, fileList)):
    for match, filename in zip(matchList, fileList):
        if not match:
            continue
        match.group(1)
        if match.group(2):
            first = True
            benchName = match.group(3)
        else:
            # print("Not First")
            first = False
            benchName = match.group(4)
            # print("Set benchName: ", benchName)

        print("Match for {} file is {} and {}".format(
            filename, match.group(2), match.group(5)))

        if benchName in graphs:
            graph = graphs[benchName]
        else:
            graph = PyTorchGeometricTrain.TrainGraph(benchName)
            graphs[benchName] = graph

        if match.group(5) == "edgelist" and first is False:
            # Process edgeList File
            print("parsing: ", filename)
            parse_edgeList_file(filename, graph)
        elif match.group(5) == "historycosts":
            # Process historycosts File
            parse_historycosts_file(filename, graph, first=first)

    outputFirstLastCSV(outputDirectory, graphs)


def FindSpecificFiles(directory, extension):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    directory : [type]
        [description]
    extension : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """    
    # print("Called FindSpecificFiles in drectory: ", directory, "
    # with extension: ", extension)
    # print("glob was sent...", os.path.join(directory, extension))
    # print("glob returns...", glob.glob(os.path.join(directory, extension)))
    # return glob.glob(os.path.join(directory, extension))
    # print(os.listdir(directory))
    return [f for f in os.listdir(directory) if f.endswith(extension)]

# @lru_cache(maxsize=32)


def parse_edge_features(f):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    f : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    with open(f) as cF:
        start_time = time.time()
        df = pd.read_csv(f)
        # print("--- Converting Edge CSV to Dataframe took %s seconds ---" % (time.time() - start_time))
        edge_index = [[], []]
        start_time = time.time()   
        edge_index[0] = df['src_node'].values
        edge_index[1] = df['sink_node'].values
        # print("--- Handling Edges took %s seconds ---" % (time.time() - start_time))
        
    return torch.tensor(edge_index, dtype=torch.long)


    
def parse_node_features(f, g):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    f : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """    
    # match = re.search(CSV_FILE_STRING, f)
    # if not match:
    #     return None
    # match.group(1)
    # graph = PyTorchGeometricTrain.TrainGraph(benchName)
    start_time = time.time()
    # Features
    with open(f) as cF:
        df = pd.read_csv(cF)
        # Grab from a separate CSV
        # Parse Node Features
        df = df.drop(['node_id'], axis=1)

        one_hot = pd.get_dummies(df['node_type'])
        df = df.drop(['node_type'], axis=1)
        df = df.join(one_hot)
        # ! We hate these features. DELETE THEM
        df = df.drop(['src_node', 'sink_node'], axis=1)
        df = df.apply(pd.to_numeric)
        # df = (df-df.min())/(df.max()-df.min())
        # x = np.nan_to_num(df.values)
        x = df.values
        in_netlist = df['in_netlist']
    # Target
    with open(g) as cG:
        df = pd.read_csv(cG)
        df['history_cost'] = df['history_cost'] - 1
        y = list(df['history_cost'].values)
        y = [[i] for i in y]
        # df = [df.in_netlist == 1]
    # print("--- Handling Nodes took %s seconds ---" % (time.time() - start_time))   
    # print("---- Processed in %.2f seconds ----" % (time.time() - start_time))
    return torch.tensor(x, dtype=torch.float),\
        torch.tensor(y, dtype=torch.float), in_netlist



def parse_one_first_last_csv_old(f):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    f : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """    
    match = re.search(CSV_FILE_STRING, f)
    if not match:
        return None
    benchName = match.group(1)
    graph = PyTorchGeometricTrain.TrainGraph(benchName)
    with open(f) as cF:
        reader = csv.DictReader(cF)
        lines = [row for row in reader]

        try:
            for row in lines:
                graph.NodeFromDict(row)
        except KeyError:
            print("KeyError on row: ", row)
            exit(1)
    return graph


def parse_first_last_csv(files):
    graphs = dict()

    for f in files:
        match = re.search(CSV_FILE_STRING, f)
        if not match:
            continue
        benchName = match.group(1)
        graph = PyTorchGeometricTrain.TrainGraph(benchName)
        graphs[benchName] = graph
        with open(f) as cF:
            reader = csv.DictReader(cF)
            lines = [row for row in reader]

            try:
                for row in lines:
                    graph.NodeFromDict(row)
            except KeyError:
                print("KeyError on row: ", row)
                exit(1)
    return graphs


def parse_edgeList_file(csvFile, graph):

    with open(csvFile) as cF:
        reader = csv.DictReader(cF)
        lines = [row for row in reader]

        for row in lines:
            node_id = row["src_node"]
            sink_node = row["sink_node"]
            if sink_node == 0 or sink_node == "0":
                print("Row found: ", row)
            graph.SafeAddEdge(node_id, sink_node)


def parse_historycosts_file(csvFile, graph, first=False):

    with open(csvFile) as cF:
        reader = csv.DictReader(cF)
        lines = [row for row in reader]
        # print(lines)
        # print(csvFile)

        for row in lines:
            try:
                node_id = row["Node_ID"]
            except KeyError:
                print("Node_ID not found... has: ",
                      [key for key in row.keys()])
                exit(1)
            history_cost = row["History_Cost"]
            if first:
                graph.SafeAddPrevHistory(node_id, history_cost)
            else:
                graph.SafeAddTargetHistory(node_id, history_cost)


def outputFirstLastCSV(outputDirectory, graphs):
    for graphID in graphs:
        graph = graphs[graphID]
        bench = graph.GetBenchName()
        csv_file = os.path.join(outputDirectory, bench+".csv")
        with open(csv_file, 'w+') as cF:
            writer = csv.DictWriter(cF, fieldnames=graph.GetKeys())
            writer.writeheader()
            nodeDict = graph.GetNodes()
            for node in nodeDict:
                writer.writerow(nodeDict[node].ToDict())


def outputCSV(bench, output_data):
    csv_file = bench + ".csv"
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['name', 'nodes', 'edges'])
        writer.writeheader()
        versions = ['name', 'nodes', 'edges']
        for key in output_data:
            {field: output_data[key].get(
                field) or "EMPTY" for field in versions}
            writer.writerow({field: output_data[key].get(
                field) or key for field in versions})


def outputGraphCSV(output_graph):
    csv_file = output_graph.GetBenchName() + ".csv"
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=output_graph.GetKeys())
        writer.writeheader()
        nodeDict = output_graph.GetNodes()
        for node in nodeDict:
            writer.writerow(nodeDict[node].ToDict())


def ParseHistoryCSV(CSVFile, bench_name):
    with open(CSVFile) as cFile:
        reader = csv.reader(cFile)
        lines = [row for row in reader]

    topRow = None

    graph = PyTorchGeometricTrain.TrainGraph(bench_name)

    for line in lines:
        if line[0] == "Node_ID":
            topRow = line
        elif topRow is not None:
            for element, column in zip(line, topRow):
                if column == "Node_ID":
                    node_id = element
                elif column == "History_Cost":
                    history_cost = element
            graph.AddNode(node_id, history_cost)
    return graph


def parseGraph(rr_graph_edges):
    # # Create NetworkX Directed Graph
    # # G = nx.DiGraph()
    # # Add Edges
    # G.add_edges_from(rr_graph_edges)
    # # Draw Graph && Save to png
    # # nx.draw(G, with_labels=True, font_weight='bold')
    # # plt.savefig("test.png")
    # return G
    pass


def main(options):
    # ! Update to take in input from dir
    # Parse xml file into 2 lists.
    # parseXML_SAX('/benchmarks/XML/tseng_EArch.xml')
    # rr_graph_edges = parseXML_SAX('test')
    # G = parseGraph()
    # Print Edge-List to file
    # nodes, edges = parseXML_SAX_Metrics(os.getcwd()+
    # '/benchmarks/XML/tseng_EArch')

    # output_data = collect_graph_info(options.inputDirectory)
    # outputCSV(dir1,output_data)

    # collect_graph_edges(options.inputDirectory,
    #                     options.historyCostDirectory,
    #                     options.outputDirectory)
    parse_first_last_files(options.inputDirectory,
                           options.outputDirectory)

    # output_data = collect_graph_info("/benchmarks/EARCH_MCNC/")
    # outputCSV(dir1,output_data)
    # output_data = collect_graph_info("/benchmarks/STRATXIV_MCNC/")
    # outputCSV(dir2,output_data)
    # # output_data = collect_graph_info("/benchmarks/STRATXIV_TITAN/")
    # # outputCSV(dir3,output_data)
    # output_data = collect_graph_info("/benchmarks/STRATXIV_TITANJR/")
    # outputCSV(dir4,output_data)


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
