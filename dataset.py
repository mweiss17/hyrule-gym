""" pre-process and write the labels, spatial graph, and lower resolution images to disk """
from __future__ import print_function, division
import glob
import collections
import xml.etree.ElementTree as et
from tqdm import tqdm
import networkx as nx
import pandas as pd
import numpy as np
import cv2

def process_labels(region):
    """ This function processes the labels into a nice format for the simulator"""
    labels = []
    for path in glob.glob("data/" + region + "/raw/labels/*.xml"):
        xtree = et.parse(path)
        xroot = xtree.getroot()
        for node in xroot:
            if node.tag != "object":
                continue
            frame = path.split("_")[-1].split(".")[0]
            name = node.find("name").text
            obj_type, val = name.split("-")
            bndbox = (node.find("bndbox").find("xmin").text, node.find("bndbox").find("xmax").text, node.find("bndbox").find("ymin").text, node.find("bndbox").find("ymax").text)
            labels.append((int(frame) - 1, obj_type, val, bndbox))
    label_df = pd.DataFrame(labels)
    label_df.columns = ["frame", "obj_type", "val", "coords"]
    return label_df

def construct_spatial_graph(data_df, label_df, region="saint-urbain"):
    """ Filter the pano coordinates by spatial relation and write the filtered graph to disk"""
    # Init graph
    G = nx.Graph()
    G.add_nodes_from(data_df.index.values.tolist())
    nodes = G.nodes
    max_node_distance = 1.5

    for node_1 in nodes:
        # write pano metadata to nodes
        meta = data_df.loc[node_1]
        coords1 = np.array([meta['x'], meta['y'], meta['z']])
        G.nodes[node_1]['coords'] = coords1
        G.nodes[node_1]['frame'] = meta.name
        G.nodes[node_1]['angle'] = meta.angle

        # Check the distance between the nodes
        for node_2 in G.nodes:
            if node_1 == node_2:
                continue
            meta2 = data_df.loc[node_2]
            coords2 = np.array([meta2['x'], meta2['y'], meta2['z']])
            G.nodes[node_2]['coords'] = coords2
            node_distance = np.linalg.norm(coords1 - coords2)
            if node_distance > max_node_distance:
                continue
            G.add_edge(node_1, node_2, weight=node_distance)

    # find target panos -- they are the ones with the biggest bounding box around the house number
    goal_panos = {}
    for house_number in label_df[label_df.obj_type == "house_number"]["val"].unique():
        house_number = str(house_number)
        matches = label_df[label_df.val == house_number]
        areas = []
        for coords in [x for x in matches['coords'].values]:
            areas.append((int(coords[1]) - int(coords[0])) * (int(coords[3]) - int(coords[2])))
        goal_pano = matches.iloc[areas.index(max(areas))]
        goal_panos[goal_pano["frame"]] = goal_pano

    for node in nodes:
        G.nodes[node]['goals_achieved'] = []

    for node in nodes:
        if node in goal_panos.keys():
            G.nodes[node]['goals_achieved'].append(int(goal_panos[node]["val"]))

    nx.write_gpickle(G, "data/" + region + "/processed/graph.pkl")
    return G

def create_dataset(region="saint-urbain", limit=None):
    """
    Loads in the pano images from disk, crops them, resizes them, and writes them to disk.
    Then pre-processes the pose data associated with the image and calls the fn to create the graph and to process the labels
    """
    label_df = process_labels(region)

    #get png names and apply limit
    paths = glob.glob("data/" + region + "/panos/*.png")
    if limit:
        paths = paths[:limit]

    # do electronic stabilization and resize the images
    height = 126
    width = 224
    crop_margin = int(height * (1/6))

    thumbnails = {}
    if limit:
        coords = np.load("data/" + region + "/processed/pos_ang.npy")[:limit]
    else:
        coords = np.load("data/" + region + "/processed/pos_ang.npy")

    # Get panos and crop'em into thumbnails
    for path in tqdm(paths, desc="Loading thumbnails"):
        frame = int(path.split("_")[-1].split(".")[0]) - 1
        image = cv2.imread(path)
        shape = image.shape[0:2]
        image = cv2.resize(image, (width, height))[:, :, ::-1]
        image = image[crop_margin:height - crop_margin]
        thumbnails[frame] = image

        # change label coords to mini space
        labels = label_df[label_df["frame"] == frame]
        if labels.any().any():
            for idx, row in labels.iterrows():
                row_coords = [int(x) for x in row["coords"]]
                new_coords = (int(width * row_coords[0] / shape[1]),
                              int(width * row_coords[1] / shape[1]),
                              int((height - 2 * crop_margin) * row_coords[2] / shape[0]),
                              int((height - 2 * crop_margin) * row_coords[3] / shape[0]))
                label_df.at[idx, "coords"] = new_coords
    label_df.to_hdf("data/" + region + "/processed/labels.hdf5", key="df", index=False)

    od = collections.OrderedDict(sorted(thumbnails.items()))
    coords = coords[coords[:, 0].argsort()]
    coords[:, 0] = np.array([int(x) - 1 for x in coords[:, 0]])
    intersection = set(coords[:, 0]).intersection(set(od.keys()))

    to_keep = list()
    for idx, frame in enumerate(coords[:, 0]):
        if frame in intersection:
            to_keep.append(idx)
    coords = np.take(coords, to_keep, axis=0)

    to_delete = list()
    for key in thumbnails.keys():
        if key not in intersection:
            to_delete.append(key)

    for key in to_delete:
        del od[key]

    x = coords[:, 1]
    y = coords[:, 2]
    z = coords[:, 3]
    angle = coords[:, 4]

    data_df = pd.DataFrame({"x": x, "y": y, "z": z, "angle": angle, "thumbnail": list(od.values())})
    data_df.index = coords[:, 0]
    data_df.to_hdf("data/" + region + "/processed/data.hdf5", key="df", index=False)
    construct_spatial_graph(data_df, label_df, region)
