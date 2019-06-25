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
import gzip
import pickle
from matplotlib import pyplot as plt

height = 126
width = 224
shape = (3840, 2160)
crop_margin = int(height * (1/6))

def process_labels(paths, G, coords):
    """ This function processes the labels into a nice format for the simulator"""
    labels = []
    for p in paths:
        xtree = et.parse(p)
        xroot = xtree.getroot()
        for node in xroot:
            if node.tag != "object":
                continue
            frame = int(p.split("_")[-1].split(".")[0])
            name = node.find("name").text
            obj_type, val = name.split("-")
            bndbox = (node.find("bndbox").find("xmin").text, node.find("bndbox").find("xmax").text, node.find("bndbox").find("ymin").text, node.find("bndbox").find("ymax").text)
            labels.append((frame, obj_type, val, bndbox))
    label_df = pd.DataFrame(labels, columns = ["frame", "obj_type", "val", "coords"])
    # change label coords to mini space
    label_df = label_df[label_df.frame.isin(coords)]
    labels = label_df[label_df["frame"] == int(frame)]

    if labels.any().any():
        for ix, row in labels.iterrows():
            row_coords = [int(x) for x in row["coords"]]
            new_coords = (int(width * row_coords[0] / shape[1]),
                          int(width * row_coords[1] / shape[1]),
                          int((height - 2 * crop_margin) * row_coords[2] / shape[0]),
                          int((height - 2 * crop_margin) * row_coords[3] / shape[0]))
            label_df.at[ix, "coords"] = new_coords

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

    for node in G.nodes:
        G.nodes[node]['goals_achieved'] = []

    for node in G.nodes:
        frame = int(G.nodes[node]['timestamp'].values[0] * 30)
        if frame in goal_panos.keys():
            G.nodes[node]['goals_achieved'].append(int(goal_panos[frame]["val"]))

    return label_df, G

def construct_spatial_graph(coords_df, node_blacklist, edge_blacklist, add_edges, path, mini_corl=False):
    """ Filter the pano coordinates by spatial relation and write the filtered graph to disk"""
    # Init graph
    G = nx.Graph()
    coords_df = coords_df[~coords_df.index.isin(node_blacklist)]
    if mini_corl:
        box = (24, 76, -125, 10)
        coords_df = coords_df[((coords_df.x > box[0]) & (coords_df.x < box[1]) & (coords_df.y > box[2]) & (coords_df.y < box[3]))]
    G.add_nodes_from(coords_df.index)

    nodes = G.nodes
    for node_1_idx in tqdm(nodes, desc="Adding edges to graph"):
        meta = coords_df[coords_df.index == node_1_idx]
        coords = np.array([meta['x'].values[0], meta['y'].values[0], meta['z'].values[0]])
        G.nodes[node_1_idx]['coords'] = coords
        G.nodes[node_1_idx]['timestamp'] = meta.timestamp
        G.nodes[node_1_idx]['angle'] = meta.angle

        radius = 1.1
        nearby_nodes = coords_df[(coords_df.x > coords[0] - radius) & (coords_df.x < coords[0] + radius) & (coords_df.y > coords[1] - radius) & (coords_df.y < coords[1] + radius)]
        for node_2_idx, node_2_vals in nearby_nodes.iterrows():
            if node_1_idx == node_2_idx:
                continue
            meta2 = coords_df[coords_df.index == node_2_idx]
            coords2 = np.array([meta2['x'].values[0], meta2['y'].values[0], meta2['z'].values[0]])
            G.nodes[node_2_idx]['coords'] = coords2
            node_distance = np.linalg.norm(coords - coords2)
            G.add_edge(node_1_idx, node_2_idx, weight=node_distance)

    for n1, n2 in edge_blacklist:
        if n1 in G.nodes and n2 in G.nodes:
            G.remove_edge(n1, n2)

    for n1, n2 in add_edges:
        if n1 in G.nodes and n2 in G.nodes:
            G.add_edge(n1, n2)

    return G, coords_df

def process_images(paths):
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

    images = {frame: img for frame, img in zip(frames, thumbnails)}
    return images

def construct_graph_cleanup(mini_corl=False):
    node_blacklist = [928, 929, 930, 931, 1138, 6038, 6039, 5721, 5722, 6091, 6090, 6039, \
                      6082, 6197, 6039, 6088, 4809, 5964, 5504, 5505, 5467, 5514, 174,    \
                      188, 189, 190, 2390, 2391, 2392, 2393, 1862, 1863, 1512, 1821, 4227,\
                      1874, 3894, 3895, 3896, 3897, 3898]
    node_blacklist.extend([x for x in range(1330, 1346)])
    node_blacklist.extend([x for x in range(3034, 3071)])
    node_blacklist.extend([x for x in range(879, 892)])
    node_blacklist.extend([x for x in range(2971, 2983)])
    node_blacklist.extend([x for x in range(2888, 2948)])
    node_blacklist.extend([x for x in range(2608, 2629)])
    node_blacklist.extend([x for x in range(3091, 3098)])
    node_blacklist.extend([x for x in range(704, 780)])
    node_blacklist.extend([x for x in range(118,128)])
    node_blacklist.extend([x for x in range(5724, 5748)])
    node_blacklist.extend([x for x in range(6186, 6197)])
    node_blacklist.extend([x for x in range(4891, 4896)])
    node_blacklist.extend([x for x in range(6083, 6088)])
    node_blacklist.extend([x for x in range(5516, 5600)])
    node_blacklist.extend([x for x in range(5955, 5964)])
    node_blacklist.extend([x for x in range(5459, 5467)])
    node_blacklist.extend([x for x in range(3400, 3418)])
    node_blacklist.extend([x for x in range(4261, 4266)])
    node_blacklist.extend([x for x in range(5506, 5514)])
    node_blacklist.extend([x for x in range(3876, 3894)])
    node_blacklist.extend([x for x in range(5340, 5358)])
    node_blacklist.extend([x for x in range(1122, 1132)])
    node_blacklist.extend([x for x in range(652, 704)])
    node_blacklist.extend([x for x in range(3899, 4227)])
    node_blacklist.extend([x for x in range(2394, 2410)])
    node_blacklist.extend([x for x in range(585, 608)])
    node_blacklist.extend([x for x in range(2045, 2057)])
    node_blacklist.extend([x for x in range(2244, 2252)])
    node_blacklist.extend([x for x in range(2847, 2887)])
    node_blacklist.extend([x for x in range(3636, 3652)])
    node_blacklist.extend([x for x in range(2834, 2847)])
    node_blacklist.extend([x for x in range(3652, 3661)])
    node_blacklist.extend([x for x in range(4228, 4261)])
    node_blacklist.extend([x for x in range(3251, 3257)])
    node_blacklist.extend([x for x in range(3229, 3237)])
    node_blacklist.extend([x for x in range(1835, 1843)])
    node_blacklist.extend([x for x in range(5102, 5113)])
    edge_blacklist = [(913, 915), (835, 925), (824, 826), (835, 837), (900, 902), (901, 903),        \
                      (1534, 1536), (1511, 1724)]
    add_edges = [(902,1329), (893, 894), (894, 895), (651, 2970), (638, 2983), (637, 2983), 		 \
                 (2637, 3098), (2629, 3090), (2954, 3026), (3077, 3078), (3109, 3110), (2948, 2607), \
                 (0, 1722), (6161, 6162), (6147, 6146), (6172, 6173), (6174, 6175), (4465, 4906),    \
                 (6212, 6213), (4465, 4464), (4467, 4466), (6037, 5600), (5458, 3418), (5129, 5128), \
                 (100, 101), (98, 1348), (99, 1348), (3630, 3631), (3631, 3632), (3632, 3633),       \
                 (3634, 3635), (2375, 3661), (1834, 2234), (1834, 2233), (3366, 3367)]
    if mini_corl:
        node_blacklist.extend([x for x in range(877, 879)])
        node_blacklist.extend([x for x in range(52, 56)])
        node_blacklist.extend([x for x in range(31, 39)])
        node_blacklist.extend([x for x in range(2040, 2045)])
        node_blacklist.extend([x for x in range(2058, 2063)])
        node_blacklist.extend([x for x in range(3632, 3636)])
        node_blacklist.extend([x for x in range(3661, 3669)])

    return node_blacklist, edge_blacklist, add_edges

def create_dataset(data_path="/data/data/corl/", do_images=True, do_labels=True, do_graph=True, limit=None, mini_corl=False):
    """
    Loads in the pano images from disk, crops them, resizes them, and writes them to disk.
    Then pre-processes the pose data associated with the image and calls the fn to create the graph and to process the labels
    """
    data_path = os.getcwd() + data_path

    if do_graph:
        if limit:
            coords = np.load(data_path + "processed/pos_ang.npy")[:limit]
        else:
            coords = np.load(data_path + "processed/pos_ang.npy")
        coords_df = pd.DataFrame({"x": coords[:, 2], "y": coords[:, 3], "z": coords[:, 4], "angle": coords[:, -1], "timestamp": coords[:, 1], "frame": [int(x) for x in coords[:, 1]*30]})
        node_blacklist, edge_blacklist, add_edges = construct_graph_cleanup(mini_corl)

        G, coords_df = construct_spatial_graph(coords_df, node_blacklist, edge_blacklist, add_edges, data_path, mini_corl)
        coords_df.to_hdf(data_path + "processed/coords.hdf5", key='df')
        pos = {k: v.get("coords")[0:2] for k, v in G.nodes(data=True)}
        nx.draw_networkx(G, pos,
                         nodelist=G.nodes,
                         node_color='r',
                         node_size=10,
                         alpha=0.8,
                         with_label=True)
        plt.axis('equal')
        plt.show()
        nx.write_gpickle(G, data_path + "processed/graph.pkl")

    else:
        coords_df = pd.read_hdf(data_path + "processed/coords.hdf5", key="df", index=False)
        G = nx.read_gpickle(data_path + "processed/graph.pkl")

    # Get png names and apply limit
    img_paths = [data_path + "panos/pano_" + str(frame).zfill(6) + ".png" for frame in coords_df["frame"].tolist()]
    if limit:
        img_paths = img_paths[:limit]

    if do_images:
        images = process_images(img_paths)
        f = gzip.GzipFile(data_path + "processed/images.pkl.gz", "w")
        pickle.dump(images,f)
        f.close()

    if do_labels:
        label_paths = [data_path + "raw/labels/pano_" + str(frame).zfill(6) + ".xml" for frame in coords_df["frame"].tolist()]
        label_paths = [path for path in label_paths if os.path.isfile(path)]
        label_df, G = process_labels(label_paths, G, coords_df.frame.values.tolist())
        label_df.to_hdf(data_path + "processed/labels.hdf5", key="df", index=False)
        nx.write_gpickle(G, data_path + "processed/graph.pkl")



create_dataset(data_path="/data/data/mini-corl/", do_images=True, do_labels=True, do_graph=True, mini_corl=True)
