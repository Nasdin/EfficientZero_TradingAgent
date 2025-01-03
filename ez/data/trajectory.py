# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import copy
import torch
import numpy as np
import ray


class GameTrajectory:
    def __init__(self, **kwargs):
        self.obs_lst = []
        self.reward_lst = []
        self.policy_lst = []
        self.action_lst = []
        self.pred_value_lst = []
        self.search_value_lst = []
        self.bootstrapped_value_lst = []
        self.mc_value_lst = []
        self.temp_obs = []
        self.snapshot_lst = []

        self.n_stack = kwargs.get('n_stack')
        self.discount = kwargs.get('discount')
        self.unroll_steps = kwargs.get('unroll_steps')
        self.td_steps = kwargs.get('td_steps')
        self.td_lambda = kwargs.get('td_lambda')
        self.obs_shape = kwargs.get('obs_shape')
        self.max_size = kwargs.get('trajectory_size')
        self.episodic = kwargs.get('episodic')
        self.GAE_max_steps = kwargs.get('GAE_max_steps')

    def init(self, init_frames):
        assert len(init_frames) == self.n_stack
        for obs in init_frames:
            self.obs_lst.append(copy.deepcopy(obs))

    def append(self, action, obs, reward):
        assert self.__len__() <= self.max_size
        self.action_lst.append(action)
        self.obs_lst.append(obs)
        self.reward_lst.append(reward)

    def pad_over(self, tail_obs, tail_rewards, tail_pred_values, tail_search_values, tail_policies):
        """To make sure the correction of value targets, we need to add (o_t, r_t, etc) from the next history block
        , which is necessary for the bootstrapped values at the end states of this history block.
        Eg: len = 100; target value v_100 = r_100 + gamma^1 r_101 + ... + gamma^4 r_104 + gamma^5 v_105,
            but r_101, r_102, ... are from the next history block.
        """
        assert len(tail_obs) <= self.unroll_steps
        assert len(tail_policies) <= self.unroll_steps

        for obs in tail_obs:
            self.obs_lst.append(copy.deepcopy(obs))

        for reward in tail_rewards:
            self.reward_lst.append(reward)

        for pred_value in tail_pred_values:
            self.pred_value_lst.append(pred_value)

        for search_value in tail_search_values:
            self.search_value_lst.append(search_value)

        for policy in tail_policies:
            self.policy_lst.append(policy)

        # calculate bootstrapped value
        self.bootstrapped_value_lst = self.get_bootstrapped_value(value_type='prediction')

    def save_to_memory(self):
        """Post processing the data when a history block is full"""
        self.obs_lst = ray.put(np.array(self.obs_lst))
        self.reward_lst = np.array(self.reward_lst)
        self.policy_lst = np.array(self.policy_lst)
        self.action_lst = np.array(self.action_lst)
        self.pred_value_lst = np.array(self.pred_value_lst)
        self.search_value_lst = np.array(self.search_value_lst)
        self.bootstrapped_value_lst = np.array(self.bootstrapped_value_lst)

    def make_target(self, index):
        assert index < self.__len__()

        target_obs = self.get_index_stacked_obs(index)
        target_reward = self.reward_lst[index:index+self.unroll_steps+1]
        target_pred_value = self.pred_value_lst[index:index+self.unroll_steps+1]
        target_search_value = self.search_value_lst[index:index+self.unroll_steps+1]
        target_bt_value = self.bootstrapped_value_lst[index:index+self.unroll_steps+1]
        target_policy = self.policy_lst[index:index+self.unroll_steps+1]

        assert len(target_reward) == len(target_pred_value) == len(target_search_value) == len(target_policy)
        return np.array(target_obs), np.array(target_reward), np.array(target_pred_value), np.array(target_search_value), np.array(target_bt_value), np.array(target_policy)

    def store_search_results(self, pred_value, search_value, policy, idx: int = None):
        if idx is None:
            self.pred_value_lst.append(pred_value)
            self.search_value_lst.append(search_value)
            self.policy_lst.append(policy)
        else:
            self.pred_value_lst[idx] = pred_value
            self.search_value_lst[idx] = search_value
            self.policy_lst[idx] = policy

    def get_gae_value(self, value_type='prediction', value_lst_input=None, index=None, collected_transitions=None):
        td_steps = 1
        assert value_type in ['prediction', 'search']
        if value_type == 'prediction':
            value_lst = self.pred_value_lst
        else:
            value_lst = self.search_value_lst

        if value_lst_input is not None:
            value_lst = value_lst_input

        traj_len = self.__len__()
        assert traj_len

        if index is None:
            td_lambda = self.td_lambda
        else:
            delta_lambda = 0.1 * (collected_transitions - index) / 3e4
            td_lambda = self.td_lambda - delta_lambda
            td_lambda = np.clip(td_lambda, 0.65, self.td_lambda)

        lens = self.GAE_max_steps
        gae_values = np.zeros(traj_len)
        for i in range(traj_len):
            delta = np.zeros(lens)
            advantage = np.zeros(lens + 1)
            cnt = copy.deepcopy(lens) - 1
            for idx in reversed(range(i, i + lens)):
                if idx + td_steps < traj_len:
                    bt_value = (self.discount ** td_steps) * value_lst[idx + td_steps]
                    for n in range(td_steps):
                        bt_value += (self.discount ** n) * self.reward_lst[idx + n]
                else:
                    steps = idx + td_steps - len(value_lst) + 1
                    bt_value = 0 if self.episodic else (self.discount ** (td_steps - steps)) * value_lst[-1]
                    for n in range(td_steps - steps):
                        assert idx + n < len(self.reward_lst)
                        bt_value += (self.discount ** n) * self.reward_lst[idx + n]

                try:
                    delta[cnt] = bt_value - value_lst[idx]
                except:
                    delta[cnt] = 0
                advantage[cnt] = delta[cnt] + self.discount * td_lambda * advantage[cnt + 1]
                cnt -= 1

            value_lst_tmp = np.concatenate((np.asarray(value_lst)[i:i + lens], np.zeros(lens-np.asarray(value_lst)[i:i + lens].shape[0])))
            gae_values_tmp = advantage[:lens] + value_lst_tmp
            gae_values[i] = gae_values_tmp[0]

        return gae_values

    def get_bootstrapped_value(self, value_type='prediction', value_lst_input=None, index=None, collected_transitions=None):
        assert value_type in ['prediction', 'search']
        if value_type == 'prediction':
            value_lst = self.pred_value_lst
        else:
            value_lst = self.search_value_lst

        if value_lst_input is not None:
            value_lst = value_lst_input

        bt_values = []
        traj_len = self.__len__()
        assert traj_len

        if index is None:
            td_steps = self.td_steps
        else:
            delta_td = (collected_transitions - index) // 3e4
            td_steps = self.td_steps - delta_td
            td_steps = np.clip(td_steps, 1, self.td_steps).astype(np.int32)

        for idx in range(traj_len):
            if idx + td_steps < len(value_lst):
                bt_value = (self.discount ** td_steps) * value_lst[idx + td_steps]
                for n in range(td_steps):
                    bt_value += (self.discount ** n) * self.reward_lst[idx + n]
            else:
                steps = idx + td_steps - len(value_lst) + 1
                bt_value = 0 if self.episodic else (self.discount ** (td_steps - steps)) * value_lst[-1]
                for n in range(td_steps - steps):
                    assert idx + n < len(self.reward_lst)
                    bt_value += (self.discount ** n) * self.reward_lst[idx + n]

            bt_values.append(bt_value)
        return bt_values

    def get_zero_obs(self, n_stack):
        """Return zero-filled observation"""
        return [np.ones(self.obs_shape, dtype=np.float32) for _ in range(n_stack)]

    def get_current_stacked_obs(self):
        """Return the current stacked observation for model inference"""
        index = len(self.reward_lst)
        frames = ray.get(self.obs_lst)[index:index + self.n_stack]
        return frames

    def get_index_stacked_obs(self, index, padding=False, extra=0):
        """Get stacked observations starting from index"""
        unroll_steps = self.unroll_steps + extra
        frames = ray.get(self.obs_lst)[index:index + self.n_stack + unroll_steps]
        if padding:
            pad_len = self.n_stack + unroll_steps - len(frames)
            if pad_len > 0:
                pad_frames = [frames[-1] for _ in range(pad_len)]
                frames = np.concatenate((frames, pad_frames))
        return frames

    def set_inf_len(self):
        self.max_size = 100000000

    def is_full(self):
        return self.__len__() >= self.max_size

    def __len__(self):
        return len(self.action_lst)
