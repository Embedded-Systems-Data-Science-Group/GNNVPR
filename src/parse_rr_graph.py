#Python code to illustrate parsing of XML files
# importing the required modules
import csv
from io import StringIO, BytesIO
# import networkx as nx
# import matplotlib.pyplot as plt
from lxml import etree
import resource
import xml.etree.ElementTree as ET  
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
def parseXML_SAX(xmlfile):
    with open('test_titan_node.txt', 'w') as filehandle:
        # tree = etree.parse(xmlfile)
        context = etree.iterparse(xmlfile, events=('end',));
        #  context = etree.iterparse(fp,)
        for action, elem in context:
            if elem.tag=='node':
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
    # parseXML_SAX('neuron_stratixiv_arch_timing.xml')
    # rr_graph_edges = parseXML_SAX('test.xml')
    # G = parseGraph()
    #Print Edge-List to file
    print("Hi")
      
if __name__ == "__main__":
  
    # calling main function
    main()