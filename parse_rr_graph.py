#Python code to illustrate parsing of XML files
# importing the required modules
import csv
import xml.etree.ElementTree as ET
import networkx as nx
import dgl 
import matplotlib.pyplot as plt
  
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
  
def parseGraph():
    # Create NetworkX Directed Graph
    G = nx.DiGraph()
    # Add Edges
    G.add_edges_from(rr_graph_edges)
    # Draw Graph && Save to png
    # nx.draw(G, with_labels=True, font_weight='bold')
    # plt.savefig("test.png")
    return G

  
      
def main():
    # Parse xml file into 2 lists.
    rr_graph_edges = parseXML('test.xml')
    
    # G = parseGraph()
    #Print Edge-List to file
    with open('test_size.txt', 'w') as filehandle:
        for t in rr_graph_edges:
            filehandle.write(' '.join(str(s) for s in t) + '\n')
      
if __name__ == "__main__":
  
    # calling main function
    main()