""" This is the simulator for NAVI project. It defines the action and observation spaces, tracks the agent's state, and specifies game logic. """
from __future__ import print_function, division
import enum
import math
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
import cv2
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

    def __init__(self, region="saint-urbain", obs_type='image', obs_shape=(84, 84, 3)):
        self.viewer = None
        self._action_set = HyruleEnv.Actions
        self.action_space = spaces.Discrete(len(self._action_set))
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        path = "/Users/martinweiss/code/academic/" # "/home/martin/"
        self.data_df = pd.read_hdf(path + "hyrule-gym/data/" + region + "/processed/data.hdf5", key='df', mode='r')
        self.label_df = pd.read_hdf(path + "hyrule-gym/data/" + region + "/processed/labels.hdf5", key='df', mode='r')
        self.G = nx.read_gpickle(path + "hyrule-gym/data/" + region + "/processed/graph.pkl")
        self.agent_pos = 0
        self.agent_dir = 0

    def turn(self, action):
        action = self._action_set(action)
        if action == self.Actions.LEFT_BIG:
            self.agent_dir -= 67.5
        if action == self.Actions.LEFT_SMALL:
            self.agent_dir -= 22.5
        if action == self.Actions.RIGHT_SMALL:
            self.agent_dir += 22.5
        if action == self.Actions.RIGHT_BIG:
            self.agent_dir += 67.5
        self.agent_dir = self.norm_angle(self.agent_dir)


    def get_angle_between_nodes(self, n1, n2):
        x = self.G.nodes[n1]['coords'][0] - self.G.nodes[n2]['coords'][0]
        y = self.G.nodes[n1]['coords'][1] - self.G.nodes[n2]['coords'][1]
        angle = math.atan2(y, x) * 180 / np.pi
        return np.abs(self.norm_angle(angle - self.agent_dir))# - 67.5

    def transition(self):
        """
        This function calculates the angles to the other panos
        then transitions to the one that is closest to the agent's current direction
        """
        neighbors = {}
        for n in [edge[1] for edge in list(self.G.edges(self.agent_pos))]:
            neighbors[n] = self.get_angle_between_nodes(n, self.agent_pos)
        if neighbors[min(neighbors, key=neighbors.get)] > 60:
            return # noop
        self.agent_pos = min(neighbors, key=neighbors.get)

    def set_difficulty(self, difficulty):
        return

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
            reward = self.compute_reward(visible_text, self.desired_goal, {})
            print("reward: " + str(reward))
        else:
            self.turn(action)
        #print(visible_text)
        print(self.agent_pos)
        print(self.agent_dir)
        self.agent_gps = self.sample_gps(self.data_df.loc[self.agent_pos])
        rel_gps = [self.target_gps[0] - self.agent_gps[0], self.target_gps[1] - self.agent_gps[1]]
        print(rel_gps)
        obs = {"image": image, "mission": self.desired_goal, "rel_gps": rel_gps, "visible_text": visible_text}
        return obs, reward, done, {}


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
        return res_img, x, w

    def get_visible_text(self, x, w):
        visible_text = {"house_numbers": [], "street_signs": []}
        pano_labels = self.label_df[self.label_df.frame == self.agent_pos]

        if not pano_labels.any().any():
            return visible_text

        for idx, row in pano_labels[pano_labels.obj_type == 'house_number'].iterrows():
            if x < row['coords'][0] and x+w > row['coords'][1]:
                visible_text["house_numbers"].append(int(row["val"]))
        return visible_text

    def sample_gps(self, groundtruth, scale=1):
        x, y = groundtruth[['x', 'y']]
        x = x + np.random.normal(loc=0.0, scale=scale)
        y = y + np.random.normal(loc=0.0, scale=scale)
        return (x, y)

    def reset(self):
        self.agent_pos = 0 #np.random.choice(self.G.nodes)
        self.agent_dir = 0 #random.uniform(0, 1)
        self.desired_goal = 6650 # np.random.choice(self.label_df.iloc[np.random.randint(0, self.label_df.shape[0])]["house_number"])
        self.agent_gps = self.sample_gps(self.data_df.loc[self.agent_pos])
        self.target_gps = self.sample_gps(self.data_df.loc[self.closest_pano(self.desired_goal)["frame"]], scale=3.0)
        print("self.target_gps: " + str(self.target_gps))
        print("self.agent_gps: " + str(self.agent_gps))
        image, x, w = self._get_image()
        return {"image": image, "achieved_goal": self.get_visible_text(x, w), "desired_goal": self.desired_goal}

    def oracular_spectacular(self):
        # prints the ideal way to navigate to the desired goal
        target = self.closest_pano(self.desired_goal)
        path = nx.shortest_path(self.G, self.agent_pos, target=target["frame"])
        actions = []
        agent_dir = self.agent_dir
        for idx, node in enumerate(path):
            if idx + 1 == len(path):
                break
            next_node = path[idx + 1]
            angle = self.get_angle_between_nodes(node, next_node)
            while np.abs(angle - agent_dir) > 30:
                if np.sign(angle - agent_dir) == -1:
                    agent_dir -= 22.5
                    actions.append(self.Actions.LEFT_SMALL)
                elif np.sign(angle - agent_dir) == 1:
                    agent_dir += 22.5
                    actions.append(self.Actions.RIGHT_SMALL)
            actions.append(self.Actions.FORWARD)
        return actions

    def closest_pano(self, house_number):
        areas = []
        house_number = str(house_number)
        matches = self.label_df[self.label_df.val == house_number]
        for coords in [x for x in matches['coords'].values]:
            areas.append((int(coords[1]) - int(coords[0])) * (int(coords[3]) - int(coords[2])))
        match = matches.iloc[areas.index(max(areas))]
        return match

    def compute_reward(self, visible_text, desired_goal, info):
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
        if desired_goal in visible_text["house_numbers"] and self.agent_pos == self.closest_pano(desired_goal).frame:
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
            'LEFT_SMALL': ord('s'),
            'FORWARD': ord('d'),
            'RIGHT_SMALL': ord('f'),
            'RIGHT_BIG': ord(' '),
            'DONE': ord('p'),
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
