""" This is the simulator for NAVI project. It defines the action and observation spaces, tracks the agent's state, and specifies game logic. """
from __future__ import print_function, division
import enum
import math
import os
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from matplotlib import pyplot as plt
import cv2
import gym
import gzip
from gym import spaces
import h5py
import pickle


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

    @classmethod
    def norm_angle(cls, x):
        # Utility function to keep some angles in the space of -180 to 180 degrees
        if x > 180:
            x = -360 + x
        elif x < -180:
            x = 360 + x
        return x

    @classmethod
    def convert_house_numbers(cls, num):
        res = np.zeros((4, 10))
        for col, row in enumerate(str(num)):
            res[col, int(row)] = 1
        return res.reshape(-1)

    def convert_street_name(self, street_name):
        res = self.label_df[self.label_df.obj_type == 'street_sign'].street_name.unique()
        res = (res == street_name).astype(int)
        return res

    def __init__(self, path="/data/data/mini-corl/processed/", obs_type='image', obs_shape=(84, 84, 3), shaped_reward=True):
        self.viewer = None
        self._action_set = HyruleEnv.Actions
        self.action_space = spaces.Discrete(len(self._action_set))
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        path = os.getcwd() + path
        #path = "/home/rogerg/Documents/autonomous_pedestrian_project/navi/hyrule-gym" + path
        f = gzip.GzipFile(path + "images.pkl.gz", "r")
        self.images_df = pickle.load(f)
        f.close()
        self.coords_df = pd.read_hdf(path + "coords.hdf5", key='df', mode='r')
        self.label_df = pd.read_hdf(path + "labels.hdf5", key='df', mode='r')
        self.G = nx.read_gpickle(path + "graph.pkl")

        self.curriculum_learning = True
        self.agent_loc = 191 #np.random.choice(self.coords_df.index)
        self.agent_dir = 0
        self.difficulty = 0
        self.weighted = True

        self.shaped_reward = shaped_reward
        self.max_num_steps = 10000
        self.num_steps_taken = 0

    def turn(self, action):
        action = self._action_set(action)
        if action == self.Actions.LEFT_BIG:
            self.agent_dir += 67.5
        if action == self.Actions.LEFT_SMALL:
            self.agent_dir += 22.5
        if action == self.Actions.RIGHT_SMALL:
            self.agent_dir -= 22.5
        if action == self.Actions.RIGHT_BIG:
            self.agent_dir -= 67.5
        self.agent_dir = self.norm_angle(self.agent_dir)


    def get_angle_between_nodes(self, n1, n2, use_agent_dir=True):
        x = self.G.nodes[n1]['coords'][0] - self.G.nodes[n2]['coords'][0]
        y = self.G.nodes[n1]['coords'][1] - self.G.nodes[n2]['coords'][1]
        angle = (math.atan2(y, x) * 180 / np.pi) + 180
        if use_agent_dir:
            return np.abs(self.norm_angle(angle - self.agent_dir))
        else:
            return angle

    def select_goal(self, difficulty=0):
        goals = self.label_df[self.label_df.is_goal == True]
        goal = goals.loc[np.random.choice(goals.index.values.tolist())]
        goal_idx = self.coords_df[self.coords_df.frame == goal.frame].index.values[0]
        self.goal_id = goal.house_number
        label = self.label_df[self.label_df.frame == int(self.coords_df.loc[goal_idx].frame)]
        pano_rotation = self.norm_angle(self.coords_df.loc[goal_idx].angle )
        label_dir = self.norm_angle(360 * ((int(label["coords"].values[0][0]) + int(label["coords"].values[0][1])) / 2) / 224)
        goal_dir = self.norm_angle(-label_dir + pano_rotation)

        # randomly selects a node n-transitions from the goal node
        if difficulty == 0:
            nodes = [goal_idx]
        if difficulty >= 1:
            nodes = set(nx.ego_graph(self.G, goal_idx, radius=difficulty))
            nodes -= set(nx.ego_graph(self.G, goal_idx, radius=difficulty-1))
        if self.curriculum_learning:
            self.agent_loc = np.random.choice(list(nodes))
            self.agent_dir = 22.5 * np.random.choice(range(-8, 8))
        goal_address = np.append(self.convert_house_numbers(goal.house_number),  self.convert_street_name(goal.street_name))
        return goal_idx, goal_address, goal_dir


    def transition(self):
        """
        This function calculates the angles to the other panos
        then transitions to the one that is closest to the agent's current direction
        """
        neighbors = {}
        for n in [edge[1] for edge in list(self.G.edges(self.agent_loc))]:
            neighbors[n] = self.get_angle_between_nodes(n, self.agent_loc)

        if neighbors[min(neighbors, key=neighbors.get)] > 45:
            return # noop

        self.agent_loc = min(neighbors, key=neighbors.get)

    def set_difficulty(self, difficulty, weighted=False):
        self.difficulty = difficulty
        # self.weighted = weighted

    def step(self, a):
        done = False
        reward = 0.0
        action = self._action_set(a)
        image, x, w = self._get_image()
        visible_text = self.get_visible_text(x, w)

        if action == self.Actions.FORWARD:
            self.transition()
        elif action == self.Actions.DONE:
            done = True
            reward = self.compute_reward(x, {}, done)
            #print("Mission reward: " + str(reward))
        else:
            self.turn(action)

        if self.shaped_reward and action not in [self.Actions.DONE, self.Actions.NOOP]:
            reward = self.compute_reward(x, {}, done)
            #print("Current reward: " + str(reward))

        self.agent_gps = self.sample_gps(self.coords_df.loc[self.agent_loc])
        rel_gps = [self.target_gps[0] - self.agent_gps[0], self.target_gps[1] - self.agent_gps[1]]
        obs = {"image": image, "mission": self.goal_address, "rel_gps": rel_gps, "visible_text": visible_text}
        self.num_steps_taken += 1
        if self.num_steps_taken >= self.max_num_steps and done == False:
            done = True
            reward = 0.0
        s = "obs: "
        for k, v in obs.items():
            if k != "image":
                s = s + ", " + str(k) + ": " + str(v)
        # print(self.agent_loc)
        print(s)
        print(obs)
        return obs, reward, done, {}


    def _get_image(self, high_res=False, plot=False):
        if high_res:
            img = cv2.imread(filename=self.path + "/panos/pano_"+ str(self.agent_loc).zfill(6) + ".png")[:, :, ::-1]
            obs_shape = (1024, 1024, 3)
        else:
            img = self.images_df[self.coords_df.loc[self.agent_loc].frame]
            obs_shape = self.observation_space.shape

        pano_rotation = self.norm_angle(self.coords_df.loc[self.agent_loc].angle + 90)
        w = obs_shape[0]
        y = img.shape[0] - obs_shape[0]
        h = obs_shape[0]
        x = int((self.norm_angle(-self.agent_dir + pano_rotation) + 180)/360 * img.shape[1])

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
        return res_img, x, w

    def get_visible_text(self, x, w):
        visible_text = {}
        pano_labels = self.label_df[self.label_df.frame == int(self.G.nodes[self.agent_loc]['timestamp'] * 30)]
        if not pano_labels.any().any():
            return visible_text

        house_numbers = []
        for idx, row in pano_labels[pano_labels.obj_type == 'house_number'].iterrows():
            if x < row['coords'][0] and x+w > row['coords'][1]:
                house_numbers.append(self.convert_house_numbers(row["house_number"]))
        temp = np.zeros(120)
        if len(house_numbers) != 0:
            nums = np.hstack(house_numbers)[:120]
            temp[:nums.size] = nums
        visible_text["house_numbers"] = temp

        num_streets = self.label_df[self.label_df.obj_type == 'street_sign'].street_name.unique().size
        street_signs = []
        for idx, row in pano_labels[pano_labels.obj_type == 'street_sign'].iterrows():
            if x < row['coords'][0] and x+w > row['coords'][1]:
                street_signs.append(self.convert_street_name(row["street_name"]))
        temp = np.zeros(6)
        if len(street_signs) != 0:
            nums = np.hstack(street_signs)[:6]
            temp[:nums.size] = nums
        visible_text["street_names"] = temp
        return visible_text

    def sample_gps(self, groundtruth, scale=1):
        x, y = groundtruth[['x', 'y']]
        if type(x) == pd.core.series.Series:
            x = x.values[0]
            y = y.values[0]
        x = x + np.random.normal(loc=0.0, scale=scale)
        y = y + np.random.normal(loc=0.0, scale=scale)
        gps_scale = 100.0  # TODO: Arbitrary. Need a better normalizing value here. Requires min-max from dataframe.
        return (x/gps_scale, y/gps_scale)

    def reset(self):
        self.num_steps_taken = 0
        self.goal_idx, self.goal_address, self.goal_dir = self.select_goal(self.difficulty)
        self.agent_gps = self.sample_gps(self.coords_df.loc[self.agent_loc])
        self.target_gps = self.sample_gps(self.coords_df.loc[self.goal_idx], scale=3.0)
        image, x, w = self._get_image()
        rel_gps = [self.target_gps[0] - self.agent_gps[0], self.target_gps[1] - self.agent_gps[1]]
        return {"image": image, "mission": self.goal_address, "rel_gps": rel_gps, "visible_text": self.get_visible_text(x, w)}

    def angles_to_turn(self, cur, target):
        go_left = []
        go_right = []
        temp = cur
        while np.abs(target - cur) > 22.5:
            cur = (cur - 22.5) % 360
            go_right.append(self.Actions.RIGHT_SMALL)
        cur = temp
        while np.abs(target - cur) > 22.5:
            cur = (cur + 22.5) % 360
            go_left.append(self.Actions.LEFT_SMALL)
        if len(go_left) > len(go_right):
            return go_right
        return go_left

    def shortest_path_length(self):
        # finds a minimal trajectory to navigate to the target pose
        # target_index = self.coords_df[self.coords_df.frame == int(target_node_info['timestamp'] * 30)].index.values[0]
        cur_node = self.agent_loc
        cur_dir = self.agent_dir + 180
        target_node = self.goal_idx
        path = nx.shortest_path(self.G, cur_node, target=target_node)
        actions = []
        for idx, node in enumerate(path):

            if idx + 1 != len(path):
                target_dir = self.get_angle_between_nodes(node, path[idx + 1], use_agent_dir=False)
                actions.extend(self.angles_to_turn(cur_dir, target_dir))
                cur_dir = target_dir
                actions.append(self.Actions.FORWARD)
            else:
                actions.extend(self.angles_to_turn(cur_dir, self.goal_dir + 180))
                actions.append(self.Actions.DONE)
        # print(actions)
        return actions


    def compute_reward(self, x, info, done):
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
        if self.shaped_reward and not done:
            cur_spl = len(self.shortest_path_length())
            #print("SPL:", cur_spl)
            return 1.0/(2*cur_spl)
        else:
            label = self.label_df[(self.label_df.frame == self.coords_df.loc[self.goal_idx].frame) & (self.label_df.obj_type == "door") & (self.label_df.house_number == self.goal_id)]
            is_in_correct_pano = (label.frame == int(self.coords_df.loc[self.agent_loc].frame)).values[0]
            is_facing_correct_dir = False
            coords = label.coords.values[0]
            if x < coords[0] and x + 84 > coords[1]:
                is_facing_correct_dir = True
            if is_in_correct_pano and is_facing_correct_dir:
                print("achieved goal")
                return 1.0
        return 0.0

    def render(self, mode='human'):
        img, x, w = self._get_image()
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
            'LEFT_SMALL': ord('q'),
            'FORWARD': ord('w'),
            'RIGHT_SMALL': ord('e'),
            'RIGHT_BIG': ord('d'),
            'DONE': ord('s'),
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
