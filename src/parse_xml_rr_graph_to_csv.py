#Python code to illustrate parsing of XML files
# importing the required modules
import ast
import csv, re
from io import StringIO, BytesIO
# import networkx as nx
# import matplotlib.pyplot as plt
from optparse import OptionParser

import torch
from lxml import etree
#import resource
import os
import glob
import xml.etree.ElementTree as ET
from progress.bar import Bar
#from PyTorchGeometricTrain import TrainGraph
import PyTorchGeometricTrain

BENCH_NAME_STRING = r"\\([0-9A-Za-z]+)_[0-9A-Za-z]+.xml"
FIRST_LAST_PARSE_STRING = r"([0-9A-Za-z]+)_(?:(first)_([0-9A-Za-z\.]+)|([0-9A-Za-z\.]+))_([0-9A-Za-z]+).csv"
CSV_FILE_STRING = r"([0-9A-Za-z]+).csv"
CACHE_CUTOFF = 28000

def parse_first_last_files(directory, outputDirectory):
    fileList = [os.path.join(directory, fileDir) for fileDir in os.listdir(directory)]
    matchList = [re.search(FIRST_LAST_PARSE_STRING, filename) for filename in os.listdir(directory)]

    graphs = dict()

    #for match, filename in Bar('Parsing...').iter(zip(matchList, fileList)):
    for match, filename in zip(matchList, fileList):
        if not match: continue
        overallBenchmark = match.group(1)
        if match.group(2):
            first = True
            benchName = match.group(3)
        else:
            # print("Not First")
            first = False
            benchName = match.group(4)
            # print("Set benchName: ", benchName)

        print("Match for {} file is {} and {}".format(filename, match.group(2), match.group(5)))

        if benchName in graphs:
            graph = graphs[benchName]
        else:
            graph = PyTorchGeometricTrain.TrainGraph(benchName)
            graphs[benchName] = graph

        if match.group(5) == "edgelist" and first == False:
            # Process edgeList File
            print("parsing: ", filename)
            parse_edgeList_file(filename, graph)
        elif match.group(5) == "historycosts":
            # Process historycosts File
            parse_historycosts_file(filename, graph, first=first)

    outputFirstLastCSV(outputDirectory, graphs)

def FindSpecificFiles(directory, extension):
    # print("Called FindSpecificFiles in drectory: ", directory, " with extension: ", extension)
    # print("glob was sent...", os.path.join(directory, extension))
    # print("glob returns...", glob.glob(os.path.join(directory, extension)))
    # return glob.glob(os.path.join(directory, extension))
    # print(os.listdir(directory))
    return [f for f in os.listdir(directory) if f.endswith(extension)]

def parse_one_first_last_csv(f):
    match = re.search(CSV_FILE_STRING, f)
    if not match: return None
    benchName = match.group(1)
    # graph = PyTorchGeometricTrain.TrainGraph(benchName)
    with open(f) as cF:
        reader = csv.DictReader(cF)
        lines = [row for row in reader]

        x = []
        y = []
        edge_index = [[], []]
        edge_index_cached = [[[], []]]
        cache_tracker = 0

        try:
            for row_dict in Bar("Parsing "+f, max=len(lines)).iter(lines):
                # graph.NodeFromDict(row)
                node_id = int(row_dict["node_id"])
                dest_edges = [int(dest) for dest in ast.literal_eval(row_dict["dest_edges"])]
                src_edges = [node_id for edge in dest_edges]
                # edge_index[0] = edge_index[0] + src_edges
                # edge_index[1] = edge_index[1] + dest_edges
                if len(edge_index_cached[cache_tracker][0]) >= CACHE_CUTOFF:
                    cache_tracker += 1
                    edge_index_cached.append([[], []])
                edge_index_cached[cache_tracker][0] += src_edges
                edge_index_cached[cache_tracker][1] += dest_edges
                x.append([float(row_dict["prev_cost"])])
                y.append([float(row_dict["history_cost"])])
            for cache in edge_index_cached:
                edge_index[0] += cache[0]
                edge_index[1] += cache[1]
        except KeyError:
            print("KeyError on row: ", row_dict)
            exit(1)
    return torch.tensor(x, dtype=torch.float),\
           torch.tensor(y, dtype=torch.float),\
           torch.tensor(edge_index, dtype=torch.long)

def parse_one_first_last_csv_old(f):
    match = re.search(CSV_FILE_STRING, f)
    if not match: return None
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
        if not match: continue
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
                print("Node_ID not found... has: ", [key for key in row.keys()])
                exit(1)
            history_cost = row["History_Cost"]
            if first: graph.SafeAddPrevHistory(node_id, history_cost)
            else: graph.SafeAddTargetHistory(node_id, history_cost)

# For the Provided directory, 
#   run all the blifs in there for each arch in arch-dir
#   savings as arch_benchmark.xml
def collect_graph_info(directory):
    info = dict()
    #directory = "/mnt/d"+directory+""
    files = glob.glob(os.path.join(directory, '*.xml'))        
    for f in files:
        bench_name = f.split('/')[-1].split('.')[0]
        print("Parsing: ", bench_name)
        nodes, edges = parseXML_SAX_Metrics(f)
        info[bench_name] = dict()
        info[bench_name]['nodes'] = nodes
        info[bench_name]['edges'] = edges
        info[bench_name]['name'] = bench_name
    return info

def collect_graph_edges(nodeDirectory, historyDircctory, outputDirectory):

    print("nodeDirectory: ", nodeDirectory)
    files = glob.glob(os.path.join(nodeDirectory, '*.xml'))

    graphs = []

    print("history directory: ", historyDircctory)
    print("Files: ", files)

    for f in files:
        match = re.search(BENCH_NAME_STRING, f)
        if not match:
            print("No bench name found for: ", f)
            continue
        else:
            bench_name = match.group(1)
        print("Parsing: ", f)

        historyFile = os.path.join(historyDircctory, bench_name+"_historycosts.csv")
        if not os.path.exists(historyFile):
            print("No history costs found for: ", bench_name)
            print("Looking for: ", historyFile)
            continue

        graph = ParseHistoryCSV(historyFile, bench_name)
        graph = parseXML_SAX_Edges(f, graph)

        outputGraphCSV(graph, outputDirectory)

    return graph

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
        writer = csv.DictWriter(csvfile,fieldnames=['name','nodes','edges'])
        writer.writeheader()
        versions = ['name','nodes','edges']
        for key in output_data:
            {field: output_data[key].get(field) or "EMPTY" for field in versions}
            writer.writerow({field: output_data[key].get(field) or key for field in versions})

def outputGraphCSV(output_graph):
    csv_file = output_graph.GetBenchName() + ".csv"
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile,fieldnames=output_graph.GetKeys())
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

def parseXML(xmlfile):
    edges = []
    # create element tree object
    tree = ET.parse(xmlfile)
    # get root element
    root = tree.getroot()
    # Get Edges as a List of Tuples
    for edge in root.iter('edge'):
        edges.append((edge.attrib['src_node'], edge.attrib['sink_node']))
    print("Parsed ", len(edges)," edges")
    return edges

def parseXML_SAX_Metrics(xmlfile):
    nodes = 0
    edges = 0 
    # tree = etree.parse(xmlfile)
    print("Checking xml file: ", xmlfile)
    context = etree.iterparse(xmlfile, events=('end',));
    #  context = etree.iterparse(fp,)
    for action, elem in context:
        if elem.tag=='node':
            nodes+=1;
        if elem.tag=='edge':
            edges+=1
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    return nodes, edges

def parseXML_SAX_Edges(xmlfile, graph):
    # tree = etree.parse(xmlfile)
    print("Checking xml file: ", xmlfile)
    context = etree.iterparse(xmlfile, events=('end',))
    print("After iterparse")
    #  context = etree.iterparse(fp,)

    for action, elem in Bar('Parsing '+xmlfile).iter(context):
        if elem.tag=='edge':
            src_node = elem.attrib.get('src_node')
            sink_node = elem.attrib.get('sink_node')
            #switch_id = elem.attrib.get('switch_id')
            graph.AddEdge(src_node, sink_node)
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    return graph

def parseXML_SAX(xmlfile):
    nodes = 0
    edges = 0
    context = etree.iterparse(xmlfile+'.xml', events=('end',));
    #  context = etree.iterparse(fp,)
    with open(xmlfile+'.txt', 'w') as filehandle:
        for action, elem in context:
            if elem.tag=='node':
                nodes+=1;
                filehandle.write(' '.join(str(s) for s in (elem.attrib['id'], elem.attrib['id'])) + '\n')
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
    
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
    # nodes, edges = parseXML_SAX_Metrics(os.getcwd()+'/benchmarks/XML/tseng_EArch')


    dir1 = "MCNC_EARCH"
    dir2 = "MCNC_TITAN"
    dir3 = "TITAN_TITAN"
    dir4 = "TITAN_JR"

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
    parser.add_option("-I", "--inputDirectory", dest="inputDirectory",
                      help="directory that contains the benchmarks to be run", metavar="INPUT")
    parser.add_option("-H", "--historyCostDirectory", dest="historyCostDirectory",
                      help="directory that contains the historyCosts to train for", metavar="HISTORY")
    parser.add_option("-O", "--outputDirectory", dest="outputDirectory",
                      help="directory that contains the combined data files for training", metavar="OUTPUT")
    (options, args) = parser.parse_args()
    # calling main function
    main(options)