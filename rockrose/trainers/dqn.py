"""
"""

from __future__ import print_function

import random

import numpy as np

import base as rr_trainer_base


class RRTrainerDQN(rr_trainer_base.RRTrainerBase):
    def __init__(self, cfg, env, model, prepr, rmem, *args, **kwargs):
        super(RRTrainerDQN, self).__init__(
            cfg, env, model, prepr, rmem, *args, **kwargs)

        self.n_actions = len(self.env._action_set)

        self.is_play = self.cfg.get('is_play', False)

        self.model_saved_file = self.cfg.get('model_saved_file', None)
        self.model_saved_per = self.cfg.get('model_saved_per', 1000)

        self.n_train_steps = self.cfg.get('n_train_steps', 3000000)

        self.if_render = self.cfg.get('if_render', False)

        self.init_epsilon = self.cfg.get('init_epsilon', 0.1)
        self.final_epsilon = self.cfg.get('final_epsilon', 0.0001)

        self.n_observe = self.cfg.get('n_observe', 3200)
        self.n_explore = self.cfg.get('n_explore', 30000)

        self.n_batch = self.cfg.get('n_batch', 32)

        self.gamma = self.cfg.get('gamma', 0.99)

        self.epsilon = self.init_epsilon

        self.t = 0

        self.init()

    def init(self):
        if self.model_saved_file:
            try:
                self.model.load(self.model_saved_file)
            except Exception as e:
                pass

    def hook_env_render(self):
        if self.if_render:
            self.env.render()

    def get_actions_greedy(self, s_t):
        a_t = np.zeros([self.n_actions])

        # choose an action epsilon greedy
        if random.random() <= self.epsilon:
            print('----rand----')
            action_index = random.randrange(self.n_actions)
            a_t[action_index] = 1
        else:
            q = self.model.predict(s_t)
            print(q)
            action_index = np.argmax(q)
            a_t[action_index] = 1

        print(' ' * 30, action_index)

        # reduced the epsilon gradually
        if self.epsilon > self.final_epsilon and self.t > self.n_observe:
            self.epsilon -= (self.init_epsilon - self.final_epsilon) / self.n_explore

        return a_t

    def train(self):

        x_t = self.env.reset()
        s_t = self.prepr.process(x_t)

        while self.t < self.n_train_steps:
            print('=' * 20, self.t)

            self.hook_env_render()

            a_t = self.get_actions_greedy(s_t)
            action_index = np.argmax(a_t)

            x_t1_colored, r_t, terminal, info = self.env.step(action_index)

            r_t = self.prepr.process_reward(r_t)
            print('r_t: ', r_t)

            if terminal:
                self.env.reset()

            s_t1 = self.prepr.process(x_t1_colored, s_t)

            if not self.is_play:
                self.rmem.append((s_t, action_index, r_t, s_t1, terminal))

            loss = 0

            # only train if done observing
            if self.t > self.n_observe and not self.is_play:
                loss += self.train_a_batch()

            s_t = s_t1

            if self.t % self.model_saved_per == 0:
                if self.model_saved_file:
                    self.model.save(self.model_saved_file)

            self.t = self.t + 1

    def train_a_batch(self):

        minibatch = self.rmem.sample(self.n_batch)

        s_t = minibatch[0][0]
        inputs = np.zeros((self.n_batch, s_t.shape[1], s_t.shape[2], s_t.shape[3]))  # 32, 80, 80, 4
        targets = np.zeros((inputs.shape[0], self.n_actions))                        # 32, 2

        for i in range(0, len(minibatch)):
            state_t = minibatch[i][0]
            action_t = minibatch[i][1]  # this is action index
            reward_t = minibatch[i][2]
            state_t1 = minibatch[i][3]
            terminal = minibatch[i][4]

            inputs[i:i + 1] = state_t

            targets[i] = self.model.predict(state_t)
            Q_sa = self.model.predict(state_t1)

            if terminal:
                # if terminated, only equals reward
                targets[i, action_t] = reward_t
            else:
                targets[i, action_t] = reward_t + self.gamma * np.max(Q_sa)

        # targets2 = normalize(targets)
        loss = self.model.train_on_batch(inputs, targets)
        return loss

    def play(self):

        while 1:

            x_t = self.env.reset()
            s_t = self.prepr.process(x_t)

            terminal = False

            while not terminal:

                self.hook_env_render()

                a_t = self.get_actions_greedy(s_t)
                action_index = np.argmax(a_t)

                x_t1_colored, r_t, terminal, info = self.env.step(action_index)

                s_t1 = self.prepr.process(x_t1_colored, s_t)

                s_t = s_t1
