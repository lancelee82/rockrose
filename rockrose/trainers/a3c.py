"""
"""

#from __future__ import print_function

import sys
import random
import time

import threading

import numpy as np

import base as rr_trainer_base


class RRTrainerA3C(rr_trainer_base.RRTrainerBase):
    def __init__(self, cfg, env, model, prepr, rmem, *args, **kwargs):
        super(RRTrainerA3C, self).__init__(
            cfg, env, model, prepr, rmem, *args, **kwargs)

        if isinstance(self.env, (list, tuple, set)):
            self.n_actions = len(self.env[0]._action_set)
        else:
            self.n_actions = len(self.env._action_set)

        self.is_play = self.cfg.get('is_play', False)

        self.thread_n = self.cfg.get('thread_n', 1)

        self.model_saved_file_p = self.cfg.get('model_saved_file_p', None)
        self.model_saved_file_v = self.cfg.get('model_saved_file_v', None)
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

        self.lock = threading.Lock()
        self.thrds = []

        self.init()

    def init(self):
        if self.model_saved_file_p:
            try:
                self.model.load(self.model_saved_file_p, self.model_saved_file_v)
            except Exception as e:
                print e


    def hook_env_render(self):
        if self.if_render:
            self.env.render()


    def get_actions_greedy(self, s_t):
        a_t = np.zeros([self.n_actions])

        # choose an action epsilon greedy
        if random.random() <= self.epsilon:
            action_index = random.randrange(self.n_actions)
            a_t[action_index] = 1
        else:
            q = self.model.predict(s_t)
            action_index = np.argmax(q)
            a_t[action_index] = 1

        # reduced the epsilon gradually
        if self.epsilon > self.final_epsilon and self.t > self.n_observe:
            self.epsilon -= (self.init_epsilon - self.final_epsilon) / self.n_explore

        return a_t


    def sample_policy_action(self, num_actions, probs):
        """
        Sample an action from an action probability distribution output by
        the policy network.
        """
        # Subtract a tiny value from probabilities in order to avoid
        # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
        probs = probs - np.finfo(np.float32).epsneg

        histogram = np.random.multinomial(1, probs)
        action_index = int(np.nonzero(histogram)[0])
        return action_index


    def train_a_thread(self, tid):
        time.sleep(tid * 1)

        # get my env
        env = self.env[tid]

        self.lock.acquire()
        x_t = env.reset()
        s_t = self.prepr.process(x_t)
        self.lock.release()

        terminal = False

        t_max = self.n_batch  #32

        while self.t < self.n_train_steps:

            s_batch = []
            past_rewards = []
            a_batch = []

            t = 0
            t_start = t

            while not (terminal or ((t - t_start) == t_max)):

                self.hook_env_render()

                self.lock.acquire()
                probs = self.model.predict_p(s_t)[0]
                self.lock.release()

                action_index = self.sample_policy_action(self.n_actions, probs)
                a_t = np.zeros([self.n_actions])
                a_t[action_index] = 1

                s_batch.append(s_t)
                a_batch.append(a_t)

                self.lock.acquire()
                x_t1_colored, r_t, terminal, info = env.step(action_index)
                s_t1 = self.prepr.process(x_t1_colored, s_t)
                self.lock.release()

                r_t = np.clip(r_t, -1, 1)
                r_t = self.prepr.process_reward(r_t)
                past_rewards.append(r_t)

                t += 1

                s_t = s_t1

                self.lock.acquire()
                print '-' * 20, tid, '    ', self.t
                print probs
                print ' ' * 40, action_index
                print 'r_t: ', r_t
                self.lock.release()


            if terminal:
                R_t = 0
            else:
                self.lock.acquire()
                # bootstrap from last state
                R_t = self.model.predict_v(s_t)[0][0]
                self.lock.release()

            R_batch = np.zeros(t)
            for i in reversed(range(t_start, t)):
                R_t = past_rewards[i] + self.gamma * R_t
                R_batch[i] = R_t

            # <3.1>  # TODO: see a3cc
            v_ts = []
            self.lock.acquire()
            for st in s_batch:
                #self.lock.acquire()
                v = self.model.predict_v(st)[0][0]
                #self.lock.release()
                v_ts.append(v)
            ##self.lock.release()

            loss = self.train_a_batch(t, s_batch, a_batch, R_batch, v_ts)
            ##self.lock.acquire()
            print '=' * 20, tid, '    ', self.t
            print 'loss: ', loss
            print 'v: ', self.model.predict_v(s_t)[0][0]
            self.lock.release()

            if terminal:
                self.lock.acquire()
                x_t = env.reset()
                s_t = self.prepr.process(x_t)
                self.lock.release()

                terminal = False

                # Reset per-episode counters
                ep_reward = 0
                ep_t = 0

            if self.t > 0 and self.t % self.model_saved_per == 0:
                if self.model_saved_file_p:
                    self.lock.acquire()
                    self.model.save(self.model_saved_file_p, self.model_saved_file_v)
                    self.lock.release()

            self.t = self.t + 1


    def train_a_batch(self, tn, s_batch, a_batch, R_batch, v_ts):
        """
        TODO: train a batch as one step -> set value needs fixed shape!
        """
        loss = 0.0

        p_targets = np.zeros((1, self.n_actions))

        for i in range(tn):
            #print i

            s_t = s_batch[i]
            inputs = s_t.reshape((1, s_t.shape[1], s_t.shape[2], s_t.shape[3]))

            # <3.2>
            #v = self.model.value_network.predict(s_t)[0][0]

            self.model.set_value(self.model.R_t, R_batch[i])
            self.model.set_value(self.model.a_t, a_batch[i])
            self.model.set_value(self.model.v_t, v_ts[i])

            v_targets = np.array([R_batch[i]])

            loss += self.model.train_on_batch_p(inputs, p_targets)
            loss += self.model.train_on_batch_v(inputs, v_targets)
        return loss


    def train(self):
        for i in range(self.thread_n):
            thrd = threading.Thread(target=self.train_a_thread, args=(i,))
            self.thrds.append(thrd)

        for t in self.thrds:
            t.start()

        while 1:
            try:
                time.sleep(60)
            except KeyboardInterrupt:
                print 'xxxxxxxxxxxxx'
                #break

                for t in self.thrds:
                    print t
                    t.join()

                sys.exit(0)


    def play(self):

        while 1:

            x_t = self.env.reset()
            s_t = self.prepr.process(x_t)

            terminal = False

            while not terminal:

                self.hook_env_render()

                probs = self.model.policy_network.predict(s_t)[0]
                action_index = self.sample_policy_action(self.n_actions, probs)

                #print s_t
                print probs
                print action_index

                x_t1_colored, r_t, terminal, info = self.env.step(action_index)
                s_t1 = self.prepr.process(x_t1_colored, s_t)

                s_t = s_t1
