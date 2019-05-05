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

class Manifest:

    def __init__(self, images_path='', hdf_path='', coords=[], dataset="Default", image_type=""):
    
        paths = glob.glob(images_path+"/*.png")
        i=0
        frames = []
        cameras = []
        x = []
        y = []
        z = []
        for path in tqdm(paths, desc="Loading Images into Memory"):
            i += 1
            
            try:
                camera = int(path.split("/")[-1].split("_")[3].split(".")[0])
                frame = int(path.split("/")[-1].split("_")[1])
                cameras.append(camera)
                frames.append(frame)
                direction = get_direction(frame)
            except IndexError:
                frames.append(-1)
                cameras.append(-1)
            
            x.append(i/8 + i % 8 + np.random.normal(loc=0.0, scale=1.0, size=None))
            y.append(i/8 + i % 8 + np.random.normal(loc=0.0, scale=1.0, size=None))
            z.append(i/8 + i % 8 + np.random.normal(loc=0.0, scale=1.0, size=None))
        self.df = pd.DataFrame({"x": x, 
                                "y": y, 
                                "z": z, 
                                "path": paths, 
                                "camera": cameras,
                                "frame": frames})
        self.df.to_hdf(hdf_path, key="df", index=False)

