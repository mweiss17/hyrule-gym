"""Stitch panoramas"""

import os
from shutil import copyfile
import networkx as nx
import matplotlib.pyplot as plt
import subprocess
import argparse

parser = argparse.ArgumentParser("Creates panoramas from vuze pictures")
parser.add_argument('--input_path', type=str, help='graph path')
args = parser.parse_args()

print("Path : " + args.input_path)
input_path = args.input_path

G = nx.read_gpickle(input_path) 
pos = {k: v.get("coords")[0:2] for k, v in G.nodes(data=True)}

# nx.draw_networkx_nodes(G, pos,
#                        nodelist=G.nodes,
#                        node_color='r',
#                        node_size=10,
#                        alpha=0.8,
#                        with_label=True)

# # First node is Green
# nx.draw_networkx_nodes(G, pos,
#                        nodelist={15},
#                        node_color='g',
#                        node_size=50,
#                        alpha=0.8)

# # Second node is blue
# nx.draw_networkx_nodes(G, pos,
#                        nodelist={16},
#                        node_color='b',
#                        node_size=50,
#                        alpha=0.8)
# edges = nx.draw_networkx_edges(G, pos=pos)

nx.draw_networkx(G, pos,
                 nodelist=G.nodes,
                 node_color='r',
                 node_size=10,
                 alpha=0.8,
                 with_label=True)

plt.axis('equal')
plt.show()