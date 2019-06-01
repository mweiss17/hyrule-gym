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
from enum import Enum
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import xml.etree.ElementTree as et


def construct_spatial_graph(data_df, region="saint-urbain"):

    # Init graph
    G = nx.Graph()
    G.add_nodes_from(data_df.index.values.tolist())
    to_remove = set(data_df.index.values.tolist())

    for n1 in G.nodes:
        meta = data_df.iloc[n1]
        coords1 = np.array([meta['x'], meta['y'], meta['z']])
        for n2 in G.nodes:
            if n1 == n2: continue
            meta2 = data_df.iloc[n2]
            coords2 = np.array([meta2['x'], meta2['y'], meta2['z']])
            node_distance = np.linalg.norm(coords1 - coords2)

    # Init vars
    max_node_distance = 1.
    for n1 in G.nodes:
        # write pano metadata to nodes
        meta = data_df.iloc[n1]
        coords1 = np.array([meta['x'], meta['y'], meta['z']])
        G.nodes[n1]['coords'] = coords1
        G.nodes[n1]['frame'] = meta.name
        G.nodes[n1]['angle'] = meta.angle

        # Check the distance between the nodes
        for n2 in G.nodes:
            if n1 == n2: continue
            if n1 in to_remove: to_remove.remove(n1)
            if n2 in to_remove: to_remove.remove(n2)

            meta2 = data_df.iloc[n2]
            coords2 = np.array([meta2['x'], meta2['y'], meta2['z']])
            G.nodes[n2]['coords'] = coords2
            node_distance = np.linalg.norm(coords1 - coords2)
            if node_distance > max_node_distance:
                continue
            G.add_edge(n1, n2, weight=node_distance)

    for n1 in to_remove:
        try:
            self.G.edges(n1)
        except Exception:
            G.remove_node(n1)

    graph_path="data/" + region + "/processed/graph.pkl"
    nx.write_gpickle(G, graph_path)
    return G

def create_dataset(region="saint-urbain", limit=None):
    dir_path="data/" + region + "/panos/"
    data_path="data/" + region + "/processed/data.hdf5"
    label_path="data/" + region + "/processed/labels.hdf5"
    label_dir_path = "data/" + region + "/raw/labels/*.xml"

    #get png names and apply limit
    paths = glob.glob(dir_path+"*.png")
    if limit:
        paths = paths[:limit]

    # Init Variables
    height = 126
    width = 224
    crop_margin = int(height * (1/6))

    i=0
    frames = []
    thumbnails = {}
    x = []
    y = []
    z = []

    path="data/saint-urbain/processed/pos_ang.npy"
    if limit:
        coords = np.load(path)[:limit]
    else:
        coords = np.load(path)

    frames = [int(x) - 1 for x in coords[:, 0]]
    x = coords[:, 1]
    y = coords[:, 2]
    z = coords[:, 3]
    angle = coords[:, 4]

    # Get panos and crop'em into thumbnails. Also makeup some fake coordinates
    for path in tqdm(paths, desc="Loading thumbnails"):
        frame = int(path.split("_")[-1].split(".")[0])
        image = cv2.imread(path)
        image = cv2.resize(image, (width, height))[:,:,::-1]
        image = image[crop_margin:height - crop_margin]
        thumbnails[frame] = image
    data_df = pd.DataFrame({"x": x, "y": y, "z": z, "angle": angle, "frame": frames, "thumbnail": list(thumbnails.values())})
    data_df.to_hdf(data_path, key="df", index=False)
    construct_spatial_graph(data_df)

    # Process the Labels
    labels = {}
    for path in glob.glob(label_dir_path):
        xtree = et.parse(path)
        xroot = xtree.getroot()
        for node in xroot:
            object_types = []
            bndboxes = []
            vals = []
            frame = path.split(" ")[1]

            if node.tag == "object":
                name = node.find("name").text
                try:
                    object_type, val = name.split("-")
                except ValueError:
                    val = None
                    object_type = name

                bndboxes.append((node.find("bndbox").find("xmin").text, node.find("bndbox").find("xmax").text, node.find("bndbox").find("ymin").text, node.find("bndbox").find("ymax").text))
                vals.append(val)
                object_types.append(object_type)
            labels[int(frame) - 1] = (int(frame) - 1, object_types, vals, bndboxes)

    label_df = pd.DataFrame(labels).T
    label_df.columns = ["frame", "obj_type", "house_number", "label_coords"]
    label_df.to_hdf(label_path, key="df", index=False)


# # This is a torch dataset version (not really used yet)
# class HyruleDataset(Dataset):
#     training_file = 'train.hdf5'
#     test_file = 'test.hdf5'
# #     classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
# #                '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
#
#     def __init__(self, root, train=True, download=False):
#         super(HyruleDataset, self)
#         self.root = root
#         self.train = train  # training set or test set
#
#         if download:
#             self.download()
#
#
#         if not self._check_exists():
#             raise RuntimeError('Dataset not found.' +
#                                ' You can use download=True to download it')
#
#         if self.train:
#             data_file = self.root + self.training_file
#         else:
#             data_file = self.root + self.test_file
#         self.df = pd.read_hdf(data_file, key='df', mode='r')
#
#     def __getitem__(self, frame):
#         """
#         Args:
#             index (int): Index
#
#         Returns:
#             tuple: (image, label) where label contains all labels for that pano.
#         """
#         image = self.df.iloc[frame]['thumbnail']
#         label = self.df.iloc[frame][["obj_type", "house_number", "label_coords"]]
#
#         return image, label
#
#     def __len__(self):
#         return len(self.data)
#
#     @property
#     def class_to_idx(self):
#         return {_class: i for i, _class in enumerate(self.classes)}
#
#     def _check_exists(self):
#         return (os.path.exists(self.root + self.training_file) )
#                 #and os.path.exists(self.test_file))
#
#     @staticmethod
#     def extract_gzip(gzip_path, remove_finished=False):
#         print('Extracting {}'.format(gzip_path))
#         with open(gzip_path.replace('.gz', ''), 'wb') as out_f, \
#                 gzip.GzipFile(gzip_path) as zip_f:
#             out_f.write(zip_f.read())
#         if remove_finished:
#             os.unlink(gzip_path)
#
#     def download(self):
#         """Download the MNIST data if it doesn't exist in processed_folder already."""
#
#         if self._check_exists():
#             return
#
#         makedir_exist_ok(self.raw_folder)
#
#         # download files
#         for url in self.urls:
#             filename = url.rpartition('/')[2]
#             file_path = os.path.join(self.raw_folder, filename)
#             download_url(url, root=self.raw_folder, filename=filename, md5=None)
#             self.extract_gzip(gzip_path=file_path, remove_finished=True)
#
#         # process and save as torch files
#         print('Processing...')
#
#         training_set = (
#             read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
#             read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
#         )
#         test_set = (
#             read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
#             read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
#         )
#         with open(self.training_file, 'wb') as f:
#             torch.save(training_set, f)
#         with open(self.test_file, 'wb') as f:
#             torch.save(test_set, f)
#
#         print('Done!')
#
#     def extra_repr(self):
#         return "Split: {}".format("Train" if self.train is True else "Test")
