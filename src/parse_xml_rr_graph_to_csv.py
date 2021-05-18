#Python code to illustrate parsing of XML files
# importing the required modules
import csv
from io import StringIO, BytesIO
# import networkx as nx
# import matplotlib.pyplot as plt
from lxml import etree
import resource
import os
import glob
import xml.etree.ElementTree as ET  

# For the Provided directory, 
#   run all the blifs in there for each arch in arch-dir
#   savings as arch_benchmark.xml
def collect_graph_info(directory):
    info = dict()
    directory = "/mnt/d"+directory+""
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

def outputCSV(bench, output_data):
    csv_file = bench + ".csv"
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile,fieldnames=['name','nodes','edges'])
        writer.writeheader()
        versions = ['name','nodes','edges']
        for key in output_data:
            {field: output_data[key].get(field) or "EMPTY" for field in versions} 
            writer.writerow({field: output_data[key].get(field) or key for field in versions})

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


def main():
    # ! Update to take in input from dir
    # Parse xml file into 2 lists.
    # parseXML_SAX('/benchmarks/XML/tseng_EArch.xml')
    # rr_graph_edges = parseXML_SAX('test')
    # G = parseGraph()
    #Print Edge-List to file
    # nodes, edges = parseXML_SAX_Metrics(os.getcwd()+'/benchmarks/XML/tseng_EArch')
    dir1 = "MCNC_EARCH"
    dir2 = "MCNC_TITAN"
    dir3 = "TITAN_TITAN"
    dir4 = "TITAN_JR"
    output_data = collect_graph_info("/benchmarks/EARCH_MCNC/")
    outputCSV(dir1,output_data)
    output_data = collect_graph_info("/benchmarks/STRATXIV_MCNC/")
    outputCSV(dir2,output_data)
    # output_data = collect_graph_info("/benchmarks/STRATXIV_TITAN/")
    # outputCSV(dir3,output_data)
    output_data = collect_graph_info("/benchmarks/STRATXIV_TITANJR/")
    outputCSV(dir4,output_data)

if __name__ == "__main__":
  
    # calling main function
    main()