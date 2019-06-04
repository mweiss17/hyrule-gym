""" This is the simulator for NAVI project. It defines the action and observation spaces, tracks the agent's state, and specifies game logic. """
from __future__ import print_function, division
import enum
import numpy as np
import pandas as pd
import networkx as nx
import cv2
import math
from matplotlib import pyplot as plt
import gym
from gym import spaces


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

    def __init__(self, region="saint-urbain", obs_type='image', obs_shape=(84, 84, 3)):
        self.viewer = None
        self._action_set = HyruleEnv.Actions
        self.action_space = spaces.Discrete(len(self._action_set))
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        self.data_df = pd.read_hdf("data/" + region + "/processed/data.hdf5", key='df', mode='r')
        self.label_df = pd.read_hdf("data/" + region + "/processed/labels.hdf5", key='df', mode='r')
        self.G = nx.read_gpickle("data/" + region + "/processed/graph.pkl")

    def norm_angle(self, x):
        if x > 180:
            x = -360 + x
        elif x < -180:
            x = 360 + x
        return x

    def turn(self, action):
        action = self._action_set(action)
        if action == self.Actions.LEFT_BIG:
            self.agent_dir -= 120
        if action == self.Actions.LEFT_SMALL:
            self.agent_dir -= 40
        if action == self.Actions.RIGHT_SMALL:
            self.agent_dir += 40
        if action == self.Actions.RIGHT_BIG:
            self.agent_dir += 120
        self.agent_dir = self.norm_angle(self.agent_dir)

    def transition(self):
        """ This function calculates the angles to the other """

        neighbors = {}
        for n in [edge[1] for edge in list(self.G.edges(self.agent_pos))]:
            x = self.G.nodes[n]['coords'][0] - self.G.nodes[self.agent_pos]['coords'][0]
            y = self.G.nodes[n]['coords'][1] - self.G.nodes[self.agent_pos]['coords'][1]
            angle = math.atan2(y, x) * 180 / np.pi
            neighbors[n] = np.abs(self.norm_angle(angle - self.agent_dir))# - 67.5

        if neighbors[min(neighbors, key=neighbors.get)] > 60:
            return # noop

        self.agent_pos = min(neighbors, key=neighbors.get)


    def step(self, a):
        action = self._action_set(a)
        if action != self.Actions.FORWARD:
            self.turn(action)
        elif action == self.Actions.FORWARD:
            self.transition()
        ob = self._get_image()

        reward = self.compute_reward(self.achieved_goal, self.desired_goal, {})
        done = False
        if reward:
            done = True
        return ob, reward, done, {'achieved_goal': self.achieved_goal}


    def _get_image(self, high_res=False, plot=False):
        if high_res:
            img = cv2.imread(filename="data/saint-urbain/panos/pano_frame_"+ str(self.agent_pos).zfill(6) + ".png")[:, :, ::-1]
            obs_shape = (1024, 1024, 3)
        else:
            img = self.data_df.loc[self.agent_pos]['thumbnail']
            obs_shape = self.observation_space.shape

        pano_rotation = self.norm_angle(self.G.node[self.agent_pos]['angle'])
        w = obs_shape[0]
        y = img.shape[0] - obs_shape[0]
        h = obs_shape[0]
        x = int((self.norm_angle(self.agent_dir + pano_rotation) + 180)/360 * img.shape[1])

        if (x + w) % img.shape[1] != (x + w):
            res_img = np.zeros(obs_shape)
            offset = img.shape[1] - (x % img.shape[1])
            res_img[:, :offset] = img[y:y+h, x:x + offset]
            res_img[:, offset:] = img[y:y+h, :(x + w) % img.shape[1]]
        else:
            res_img = img[:, x:x + w]

        if plot:
            _, ax = plt.subplots(figsize=(18, 18))
            ax.imshow(res_img.astype(int))
            plt.show()
        return res_img

    def reset(self):
        self.agent_pos = 0 #np.random.choice(self.G.nodes)
        self.agent_dir = 0 #random.uniform(0, 1)
        self.desired_goal = 1 # np.random.choice(self.label_df.iloc[np.random.randint(0, self.label_df.shape[0])]["house_number"])
        try:
            self.achieved_goal = self.label_df.loc[self.agent_pos]["house_number"]
        except Exception as e:
            self.achieved_goal = ['-1']
        return {"observation": self._get_image(), "achieved_goal": self.achieved_goal, "desired_goal": self.desired_goal}

    def get_labels(self):
        coords = label['label_coords']
        pano_width = 3840
        pano_height = 2160

        rel_coords = (int(coords[0]) / pano_width, int(coords[1]) / self.pano_width, int(coords[2]) / pano_height, int(coords[3]) / pano_height)

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
