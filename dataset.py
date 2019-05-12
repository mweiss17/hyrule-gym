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

def get_dataset(hdf_path="data/equirectangular/train.hdf5"):
    df = pd.read_hdf(hdf_path, key='df', mode='r')
    return df

def create_dataset(dir_path="data/equirectangular/",
                   data_path="data/equirectangular/data.hdf5",
                   label_path="data/equirectangular/labels.hdf5",
                   label_dir_path = "data/equirectangular/labels/*.xml", limit=None):

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
    thumbnails = []
    x = []
    y = []
    z = []

    # Get panos and crop'em into thumbnails. Also makeup some fake coordinates
    for path in tqdm(paths, desc="Loading thumbnails"):
        i += 1
        try:
            frame = path.split(" ")[-1].split(".")[0]
            frames.append(int(frame) - 1)
        except IndexError:
            frames.append(-1)
        x.append(i/8 + i % 8 + np.random.normal(loc=0.0, scale=1.0, size=None))
        y.append(i/8 + i % 8 + np.random.normal(loc=0.0, scale=1.0, size=None))
        z.append(i/8 + i % 8 + np.random.normal(loc=0.0, scale=1.0, size=None))
        image = cv2.imread(path)
        image = cv2.resize(image, (width, height))[:,:,::-1]
        image = image[crop_margin:height - crop_margin]
        thumbnails.append(image)
    data_df = pd.DataFrame({"x": x,
                            "y": y,
                            "z": z,
                            "path": paths,
                            "frame": frames,
                            "thumbnail": thumbnails}).set_index("frame").sort_values("frame")

    data_df.to_hdf(data_path, key="df", index=False)

    # Get Labels
    labels = {}
    for path in glob.glob(label_dir_path):
        frame = path.split(" ")[1]
        xtree = et.parse(path)
        xroot = xtree.getroot()
        object_types = []
        bndboxes = []
        vals = []
        for node in xroot:
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
