from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import csv
import cv2
import glob
import networkx as nx
from tqdm import tqdm
from enum import Enum, auto
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PoseGraph(object):
    def __init__(self, df):
        G = nx.Graph()
        G.add_nodes_from(df.index)
        max_node_distance = 10
        pos = {}
        edge_pos = []
        for node1 in G.nodes:
            meta = df.iloc[node1]
            coords1 = np.array([meta['x'], meta['y'], meta['z']])
            G.nodes[node1]['coords'] = coords1
            G.nodes[node1]['camera'] = meta['camera']
            G.nodes[node1]['frame'] = meta['frame'] if meta['frame'] != -1 else node1
            pos[node1] = coords1
            for node2 in G.nodes:
                if node1 == node2:
                    continue
                meta2 = df.iloc[node2]
                coords2 = np.array([meta2['x'], meta2['y'], meta2['z']])
                G.nodes[node2]['coords'] = coords2
                if np.linalg.norm(coords1 - coords2) < max_node_distance:
                    edge_pos.append((coords1, coords2))
                    G.add_edge(node1, node2, weight=np.linalg.norm(coords1-coords2))
