""" pre-process and write the labels, spatial graph, and lower resolution images to disk """
from __future__ import print_function, division
import glob
import os
import xml.etree.ElementTree as et
from tqdm import tqdm
import networkx as nx
import pandas as pd
import numpy as np
import cv2

height = 126
width = 224
shape = (3840, 2160)
crop_margin = int(height * (1/6))

def process_labels(path):
    """ This function processes the labels into a nice format for the simulator"""
    labels = []
    for p in glob.glob(path + "raw/labels/*.xml"):
        xtree = et.parse(p)
        xroot = xtree.getroot()
        for node in xroot:
            if node.tag != "object":
                continue
            frame = p.split("_")[-1].split(".")[0]
            name = node.find("name").text
            obj_type, val = name.split("-")
            bndbox = (node.find("bndbox").find("xmin").text, node.find("bndbox").find("xmax").text, node.find("bndbox").find("ymin").text, node.find("bndbox").find("ymax").text)
            labels.append((frame, obj_type, val, bndbox))
    label_df = pd.DataFrame(labels, columns = ["frame", "obj_type", "val", "coords"])
    # change label coords to mini space
    labels = label_df[label_df["frame"] == int(frame)]

    if labels.any().any():
        for ix, row in labels.iterrows():
            row_coords = [int(x) for x in row["coords"]]
            new_coords = (int(width * row_coords[0] / shape[1]),
                          int(width * row_coords[1] / shape[1]),
                          int((height - 2 * crop_margin) * row_coords[2] / shape[0]),
                          int((height - 2 * crop_margin) * row_coords[3] / shape[0]))
            label_df.at[ix, "coords"] = new_coords
    label_df.to_hdf(path + "processed/labels.hdf5", key="df", index=False)
    return label_df

def construct_spatial_graph(coords_df, label_df, path):
    """ Filter the pano coordinates by spatial relation and write the filtered graph to disk"""
    # Init graph
    G = nx.Graph()
    G.add_nodes_from(range(len(coords_df)))
    nodes = G.nodes
    max_node_distance = 1.5
    for node_1_idx in tqdm(nodes, desc="Adding edges to graph"):
        meta = coords_df[coords_df.index == node_1_idx]
        coords = np.array([meta['x'].values[0], meta['y'].values[0], meta['z'].values[0]])
        G.nodes[node_1_idx]['coords'] = coords
        G.nodes[node_1_idx]['timestamp'] = meta.timestamp
        G.nodes[node_1_idx]['angle'] = meta.angle

        radius = 0.75
        nearby_nodes = coords_df[(coords_df.x > coords[0] - radius) & (coords_df.x < coords[0] + radius) & (coords_df.y > coords[1] - radius) & (coords_df.y < coords[1] + radius)]
        for node_2_idx, node_2_vals in nearby_nodes.iterrows():
            if node_1_idx == node_2_idx:
                continue
            meta2 = coords_df[coords_df.index == node_2_idx]
            coords2 = np.array([meta2['x'].values[0], meta2['y'].values[0], meta2['z'].values[0]])
            G.nodes[node_2_idx]['coords'] = coords2
            node_distance = np.linalg.norm(coords - coords2)
            if node_distance > max_node_distance:
                continue
            G.add_edge(node_1_idx, node_2_idx, weight=node_distance)

    # find target panos -- they are the ones with the biggest bounding box an the house number
    goal_panos = {}
    for house_number in label_df[label_df.obj_type == "house_number"]["val"].unique():
        house_number = str(house_number)
        matches = label_df[label_df.val == house_number]
        areas = []
        for coords in [x for x in matches['coords'].values]:
            areas.append((int(coords[1]) - int(coords[0])) * (int(coords[3]) - int(coords[2])))
        goal_pano = matches.iloc[areas.index(max(areas))]
        goal_panos[int(goal_pano.frame)] = goal_pano

    for node in nodes:
        G.nodes[node]['goals_achieved'] = []

    for node in nodes:
        frame = int(G.nodes[node]['timestamp'].values[0] * 30)
        if frame in goal_panos.keys():
            G.nodes[node]['goals_achieved'].append(int(goal_panos[frame]["val"]))

    nx.write_gpickle(G, path + "processed/graph.pkl")
    return G

def process_images(data_path, paths):
    thumbnails = np.zeros((len(paths), 84, 224, 3))
    frames = np.zeros(len(paths))

    # Get panos and crop'em into thumbnails
    for idx, path in enumerate(tqdm(paths, desc="Loading thumbnails")):
        frame = int(path.split("_")[-1].split(".")[0])
        frames[idx] = frame
        image = cv2.imread(path)
        image = cv2.resize(image, (width, height))[:, :, ::-1]
        image = image[crop_margin:height - crop_margin]
        thumbnails[idx] = image

    images_df = {frame: img for frame, img in zip(frames, thumbnails)}
    np.savez_compressed(data_path + "processed/images.npz", images_df)


def create_dataset(data_path="/data/data/corl/", do_images=True, do_labels=True, do_graph=True, limit=None):
    """
    Loads in the pano images from disk, crops them, resizes them, and writes them to disk.
    Then pre-processes the pose data associated with the image and calls the fn to create the graph and to process the labels
    """
    data_path = os.getcwd() + data_path
    #get png names and apply limit
    paths = glob.glob(data_path + "panos/*.png")
    if limit:
        paths = paths[:limit]

    if do_images:
        process_images(data_path, paths)

    if do_labels:
        label_df = process_labels(data_path)
    else:
        label_df = pd.read_hdf(data_path + "processed/labels.hdf5", key="df", index=False)

    if do_graph:
        if limit:
            coords = np.load(data_path + "processed/pos_ang.npy")[:limit]
        else:
            coords = np.load(data_path + "processed/pos_ang.npy")
        coords_df = pd.DataFrame({"x": coords[:, 2], "y": coords[:, 3], "z": coords[:, 4], "angle": coords[:, -1], "timestamp": coords[:, 1], "frame": [int(x) for x in coords[:, 1]*30]})
        coords_df.to_hdf(data_path + "processed/coords.hdf5", key='df')
        construct_spatial_graph(coords_df, label_df, data_path)

create_dataset(data_path="/data/data/corl/", do_images=False, do_labels=False, do_graph=True)
