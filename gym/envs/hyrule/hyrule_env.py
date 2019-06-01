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
        DONE = 5

    def __init__(self, region="saint-urbain", obs_type='image'):
        self.pano_width = 3840
        self.pano_height = 2160
        self.viewer = None
        self._action_set = HyruleEnv.Actions
        self.action_space = spaces.Discrete(len(self._action_set))
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)

        data_path="data/" + region + "/processed/data.hdf5"
        label_path="data/" + region + "/processed/labels.hdf5"
        graph_path="data/" + region + "/processed/graph.pkl"
        self.data_df = pd.read_hdf(data_path, key='df', mode='r')
        self.label_df = pd.read_hdf(label_path, key='df', mode='r')
        self.G = nx.read_gpickle(graph_path)

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
        self.neighbors = [edge[1] for edge in list(self.G.edges(self.agent_pos))]
        cur_coords = self.G.nodes[self.agent_pos]['coords']
        pano_angle = self.G.nodes[self.agent_pos]['angle'] / 360

        edge_angles = {}

        # Calculate angles to each neighbor
        for n in self.neighbors:
            sink_coords = self.G.nodes[n]['coords']
            o = sink_coords[1] - cur_coords[1]
            h = np.linalg.norm(self.G.nodes[n]['coords'][0:2] - self.G.nodes[self.agent_pos]['coords'][0:2])
            edge_angle = np.arcsin(o/h)/np.pi + 0.5
            # print("edge_angle: "+str(edge_angle))
            # print("self.agent_dir: "+str(self.agent_dir))
            # print("pano_angle + edge_angle - self.agent_dir: " + str(pano_angle + edge_angle - self.agent_dir))
            edge_angles[n] = np.abs(pano_angle + edge_angle - self.agent_dir)
        if action != self.Actions.FORWARD:
            self.turn(action)
        elif action == self.Actions.FORWARD:
            if edge_angles[min(edge_angles, key=edge_angles.get)] < 60/360:
                self.agent_pos = min(edge_angles, key=edge_angles.get)
        ob = self._get_image()

        reward = self.compute_reward(self.achieved_goal, self.desired_goal, {})
        done = False
        if reward:
            done = True
        return ob, reward, done, {'achieved_goal': self.achieved_goal}

    def _get_image(self):
        image = self.data_df.iloc[self.agent_pos]['thumbnail']
        angle = self.data_df.iloc[self.agent_pos]['angle']
        x = (int(image.shape[1] * self.agent_dir) + int(image.shape[1] * angle/360)) % image.shape[1]
        w = 84

        if (x + w) % image.shape[1] != (x + w):
            img = np.zeros((84, 84, 3))

            offset = (x + w) % image.shape[1]
            offset1 = image.shape[1] - (x % image.shape[1])
            img[:, :offset1, :] = image[:, x:x + offset1]
            img[:, offset1:, :] = image[:, :offset]
            img = img.astype(int)
        else:
            img = image[:, x:x + w]
        return img

    def reset(self):
        self.agent_pos = 0#np.random.choice(self.G.nodes)
        self.agent_dir = 0#random.uniform(0, 1)
        self.desired_goal = 1# np.random.choice(self.label_df.iloc[np.random.randint(0, self.label_df.shape[0])]["house_number"])
        try:
            self.achieved_goal = self.label_df.iloc[self.agent_pos]["house_number"]
        except Exception as e:
            self.achieved_goal = ['-1']
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
        if desired_goal in achieved_goal:
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
