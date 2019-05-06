from __future__ import print_function, division
import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
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

def get_direction(i):
    if i == 0 or i == 1:
        print(Direction.FORWARD)
    if i == 2 or i == 3:
        print(Direction.RIGHT)
    if i == 4 or i == 5:
        print(Direction.LEFT)
    if i == 6 or i == 7:
        print(Direction.BACKWARD)


class Manifest:

    def __init__(self, images_path='', hdf_path='', coords=[], dataset="Default", image_type=""):
        if os.path.isfile(hdf_path):
            self.df = pd.read_hdf(hdf_path, key='df', mode='r')
            return
        paths = glob.glob(images_path+"/*.png")
        i=0
        frames = []
        cameras = []
        thumbnails = []
        x = []
        y = []
        z = []
        for path in tqdm(paths, desc="Loading thumbnails"):
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
            image = cv2.imread(path)
            image = cv2.resize(image, (0,0), fx=0.1, fy=0.1)
            thumbnails.append(image)

        self.df = pd.DataFrame({"x": x,
                                "y": y,
                                "z": z,
                                "path": paths,
                                "camera": cameras,
                                "frame": frames,
                                "thumbnail": thumbnails})
        self.df.to_hdf(hdf_path, key="df", index=False)


class HyruleEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    class Actions(enum.IntEnum):
        LEFT_BIG = 0
        LEFT_SMALL = 1
        FORWARD = 2
        RIGHT_SMALL = 3
        RIGHT_BIG = 4
        NOOP = 5

    def __init__(
            self,
            path="data/equirectangular/",
            hdf_path="data/equirectangular/manifest.hdf5",
            dataset='st_lau',
            obs_type='image'):

        self.path = path
        if not os.path.exists(self.path):
            msg = 'You asked for the env at %s but path %s does not exist'
            raise IOError(msg % self.path)
        self._obs_type = obs_type
        self.viewer = None
        self.seed()

        self._action_set = HyruleEnv.Actions
        self.action_space = spaces.Discrete(len(self._action_set))

        screen_height = 84
        screen_width = 84

        if self._obs_type == 'image':
            self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        else:
            raise error.Error('Unrecognized observation type: {}'.format(self._obs_type))

        self.m = Manifest(path, hdf_path)

        G = nx.Graph()
        G.add_nodes_from(self.m.df.index)
        max_node_distance = 20
        pos = {}
        edge_pos = []
        for node1 in G.nodes:
            meta = self.m.df.iloc[node1]
            coords1 = np.array([meta['x'], meta['y'], meta['z']])
            G.nodes[node1]['coords'] = coords1
            G.nodes[node1]['camera'] = meta['camera']
            G.nodes[node1]['frame'] = meta['frame'] if meta['frame'] != -1 else node1
            pos[node1] = coords1
            for node2 in G.nodes:
                if node1 == node2:
                    continue
                meta2 = self.m.df.iloc[node2]
                coords2 = np.array([meta2['x'], meta2['y'], meta2['z']])
                G.nodes[node2]['coords'] = coords2
                if np.linalg.norm(coords1 - coords2) < max_node_distance:
                    edge_pos.append((coords1, coords2))
                    G.add_edge(node1, node2, weight=np.linalg.norm(coords1-coords2))
        self.G = G

    def turn(self, action):
        action = self._action_set(action)
        # print(action)
        # print(self.agent_dir)
        if action == self.Actions.LEFT_BIG:
            self.agent_dir -= (1/3)
        if action == self.Actions.LEFT_SMALL:
            self.agent_dir -= (1/9)
        if action == self.Actions.RIGHT_SMALL:
            self.agent_dir += (1/9)
        if action == self.Actions.RIGHT_BIG:
            self.agent_dir += (1/3)
        self.agent_dir = self.agent_dir % 1

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        # Empirically, we need to seed before loading the ROM.
        # self.ale.setInt(b'random_seed', seed2)
        # self.ale.loadROM(self.game_path)
        return [seed1, seed2]

    def step(self, a):
        reward = 0.0
        action = self._action_set(a)

        if action != self.Actions.FORWARD:
            self.turn(action)
        elif action == self.Actions.FORWARD:
            self.edges = [edge[1] for edge in list(self.G.edges(self.agent_pos))]
            dirs = {}
            # print("my coords: " + str(self.G.nodes[self.agent_pos]['coords']))
            for e in self.edges:
                # print("e: " + str(self.G.nodes[e]['coords']) )
                a = self.G.nodes[e]['coords'][0] - self.G.nodes[self.agent_pos]['coords'][0]
                h = np.linalg.norm(self.G.nodes[e]['coords'][0:2] - self.G.nodes[self.agent_pos]['coords'][0:2])
                if np.abs(self.agent_dir - np.cos(a/h)) < 0.2:
                    self.agent_pos = e
                    break

        ob = self._get_obs()
        done = False
        return ob, reward, done, {}

    def _get_image(self):
        image = self.m.df.iloc[self.agent_pos]['thumbnail']

        x = int(image.shape[1] * self.agent_dir)
        w = int(image.shape[1] * (120/360))
        y = image.shape[0]
        h = int(image.shape[0] * (60/360))

        if (x + w) % image.shape[1] != (x + w):
            img = np.zeros((y - 2*h, w, 3))

            offset = (x + w) % image.shape[1]
            offset1 = image.shape[1] - x

            img[:y - 2*h, :offset1, :] = image[h:y - h, x:x + offset1]
            img[:y - 2*h, offset1:, :] = image[h:y - h, :offset]
        else:
            img = image[h:y - h, x:x + w]
        img = img[:,:,::-1]

        return img



    @property
    def _n_actions(self):
        return len(self._action_set)

    def _get_obs(self):
        if self._obs_type == 'image':
            img = self._get_image()
        return img

    # return: (states, observations)
    def reset(self):
        print("RESET")
        self.agent_pos = np.random.choice(len(self.G.nodes))
        self.agent_dir = random.uniform(0, 1)

        return self._get_obs()

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

    def clone_state(self):
        """Clone emulator state w/o system state. Restoring this state will
        *not* give an identical environment. For complete cloning and restoring
        of the full state, see `{clone,restore}_full_state()`."""
        state_ref = self.ale.cloneState()
        state = self.ale.encodeState(state_ref)
        self.ale.deleteState(state_ref)
        return state

    def restore_state(self, state):
        """Restore emulator state w/o system state."""
        state_ref = self.ale.decodeState(state)
        self.ale.restoreState(state_ref)
        self.ale.deleteState(state_ref)

    def clone_full_state(self):
        """Clone emulator state w/ system state including pseudorandomness.
        Restoring this state will give an identical environment."""
        state_ref = self.ale.cloneSystemState()
        state = self.ale.encodeState(state_ref)
        self.ale.deleteState(state_ref)
        return state

    def restore_full_state(self, state):
        """Restore emulator state w/ system state including pseudorandomness."""
        state_ref = self.ale.decodeState(state)
        self.ale.restoreSystemState(state_ref)
        self.ale.deleteState(state_ref)


ACTION_MEANING = {
    0: 'LEFT_BIG',
    1: 'LEFT_SMALL',
    2: 'FORWARD',
    3: 'RIGHT_SMALL',
    4: 'RIGHT_BIG',
    5: 'NOOP'
}
