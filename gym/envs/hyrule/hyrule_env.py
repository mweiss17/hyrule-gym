from __future__ import print_function, division
import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
import enum
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
import random
import xml.etree.ElementTree as et
import time

ACTION_MEANING = {
    0: 'LEFT_BIG',
    1: 'LEFT_SMALL',
    2: 'FORWARD',
    3: 'RIGHT_SMALL',
    4: 'RIGHT_BIG',
    5: 'NOOP',
    6: 'DONE'
}

class HyruleEnv(gym.GoalEnv):
    metadata = {'render.modes': ['human', 'rgb_array']}

    class Actions(enum.IntEnum):
        LEFT_BIG = 0
        LEFT_SMALL = 1
        FORWARD = 2
        RIGHT_SMALL = 3
        RIGHT_BIG = 4
        NOOP = 5
        DONE = 6

    def __init__(self, data_path="data/equirectangular/data.hdf5",
                       graph_path="data/equirectangular/graph.pkl",
                       label_path="data/equirectangular/labels.hdf5",
                       obs_type='image'):

        if not os.path.exists(data_path) or not os.path.exists(label_path):
            raise IOError("Couldn't find data or labels")

        self.pano_width = 3840
        self.pano_height = 2160
        self.viewer = None
        self._action_set = HyruleEnv.Actions
        self.action_space = spaces.Discrete(len(self._action_set))
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        self.data_df = pd.read_hdf(data_path, key='df', mode='r')
        self.label_df = pd.read_hdf(label_path, key='df', mode='r')
        self.G = self.construct_spatial_graph(graph_path)

    def construct_spatial_graph(self, graph_path):
        # Caching mechanism
        if os.path.isfile(graph_path):
            return nx.read_gpickle(graph_path)
        else:

            # Init graph
            G = nx.Graph()
            G.add_nodes_from(self.data_df.index.values.tolist())

            # Init vars
            max_node_distance = 20
            pos = {}
            edge_pos = []

            for n1 in G.nodes:
                meta = self.data_df.iloc[n1]
                coords1 = np.array([meta['x'], meta['y'], meta['z']])
                G.nodes[n1]['coords'] = coords1
                G.nodes[n1]['frame'] = meta.name
                G.nodes[n1]['house_number'] = meta['house_number']

                for n2 in G.nodes:
                    if n1 == n2: continue
                    meta2 = self.data_df.iloc[n2]
                    coords2 = np.array([meta2['x'], meta2['y'], meta2['z']])
                    G.nodes[n2]['coords'] = coords2
                    node_distance = np.linalg.norm(coords1 - coords2)
                    if node_distance > max_node_distance: continue
                    edge_pos.append((coords1, coords2))
                    G.add_edge(n1, n2, weight=node_distance)
            nx.write_gpickle(G, graph_path)
        return G

    def turn(self, action):
        action = self._action_set(action)
        if action == self.Actions.LEFT_BIG:
            self.agent_dir -= (1/3)
        if action == self.Actions.LEFT_SMALL:
            self.agent_dir -= (1/9)
        if action == self.Actions.RIGHT_SMALL:
            self.agent_dir += (1/9)
        if action == self.Actions.RIGHT_BIG:
            self.agent_dir += (1/3)
        self.agent_dir = self.agent_dir % 1

    def step(self, a):
        reward = 0.0
        action = self._action_set(a)

        if action != self.Actions.FORWARD:
            self.turn(action)
        elif action == self.Actions.FORWARD:
            self.edges = [edge[1] for edge in list(self.G.edges(self.agent_pos))]
            dirs = {}
            for e in self.edges:
                a = self.G.nodes[e]['coords'][0] - self.G.nodes[self.agent_pos]['coords'][0]
                h = np.linalg.norm(self.G.nodes[e]['coords'][0:2] - self.G.nodes[self.agent_pos]['coords'][0:2])
                if np.abs(self.agent_dir - np.cos(a/h)) < 0.2:
                    self.agent_pos = e
                    break

        ob = self._get_image()

        reward = self.compute_reward(self.achieved_goal, self.desired_goal, {})
        done = False
        if reward:
            done = True
        return ob, reward, done, {'achieved_goal': self.achieved_goal}

    def _get_image(self):
        image = self.data_df.iloc[self.agent_pos]['thumbnail']
        x = int(image.shape[1] * self.agent_dir)
        w = int(image.shape[1] * (1/3))
        h = image.shape[0]

        if (x + w) % image.shape[1] != (x + w):
            img = np.zeros((h, w, 3))

            offset = (x + w) % image.shape[1]
            offset1 = image.shape[1] - x

            img[:, :offset1, :] = image[:, x:x + offset1]
            img[:, offset1:, :] = image[:, :offset]
        else:
            img = image[:, x:x + w]
        img = cv2.resize(img, (84, 84))
        return img

    def reset(self):
        self.agent_pos = np.random.choice(len(self.G.nodes))
        self.agent_dir = random.uniform(0, 1)
        self.desired_goal = self.label_df.iloc[np.random.randint(0, self.label_df.shape[0])][["obj_type", "house_number", "label_coords"]]
        try:
            self.achieved_goal = self.label_df.iloc[self.agent_pos][["obj_type", "house_number", "label_coords"]]
        except Exception as e:
            self.achieved_goal = {"obj_type": [''], "house_number": [''], "label_coords": ['']}
        return {"observation": self._get_image(), "achieved_goal": self.achieved_goal, "desired_goal": self.desired_goal}

    def get_labels(self):
        coords = label['label_coords']

        rel_coords = (int(coords[0]) / self.pano_width, int(coords[1]) / self.pano_width, int(coords[2]) / self.pano_height, int(coords[3]) / self.pano_height)

        if not(rel_coords[0] - self.agent_dir > 0 and rel_coords[1] < self.agent_dir + (1/3)):
            label=pd.Series()
            rel_coords=None
        return label, rel_coords

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on an a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in info and compute it accordingly.

        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information

        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:

                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['goal'], info)
        """
        if achieved_goal['house_number'] == desired_goal['house_number']:
            return 1.0
        return 0.0

    def render(self, mode='human'):
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

    def get_keys_to_action(self):
        KEYWORD_TO_KEY = {
            'LEFT_BIG': ord('a'),
            'LEFT_SMALL': ord('s'),
            'FORWARD': ord('d'),
            'RIGHT_SMALL': ord('f'),
            'RIGHT_BIG': ord(' '),
            'NOOP': ord('n'),
            'DONE': ord('p'),
        }

        keys_to_action = {}

        for action_id, action_meaning in enumerate(self.get_action_meanings()):
            keys = []
            for keyword, key in KEYWORD_TO_KEY.items():
                if keyword in action_meaning:
                    keys.append(key)
            keys = tuple(sorted(keys))

            assert keys not in keys_to_action
            keys_to_action[keys] = action_id

        return keys_to_action
