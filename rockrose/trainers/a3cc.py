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
        self.tg_cfg = {}
        self.tg_cfg['flag_stop_by_ctrl_c'] = False

        self.init()

    def init(self):
        if self.model_saved_file_p:
            try:
                self.model.load(self.model_saved_file_p,
                                self.model_saved_file_v)
            except Exception as e:
                print e


    def hook_env_render(self):
        if self.if_render:
            self.env.render()


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

            # <3.1.2>
            v_ts = []

            term_batch = []

            t = 0
            t_start = t

            #while not (terminal or ((t - t_start) == t_max)):
            while not ((t - t_start) == t_max):
                #self.lock.acquire()
                if self.tg_cfg['flag_stop_by_ctrl_c']:
                    break
                #self.lock.release()

                self.hook_env_render()

                self.lock.acquire()
                #probs = self.model.policy_network.predict(s_t)[0]
                probs = self.model.predict_p(s_t)[0]
                v = self.model.predict_v(s_t)[0][0]
                self.lock.release()

                action_index = self.sample_policy_action(self.n_actions, probs)
                a_t = np.zeros([self.n_actions])
                a_t[action_index] = 1

                s_batch.append(s_t)
                a_batch.append(a_t)
                v_ts.append(v)

                self.lock.acquire()
                x_t1_colored, r_t, terminal, info = env.step(action_index)
                s_t1 = self.prepr.process(x_t1_colored, s_t)
                self.lock.release()

                #r_t = np.clip(r_t, -1, 1)
                r_t = self.prepr.process_reward(r_t)
                past_rewards.append(r_t)

                term_batch.append(terminal)

                t += 1

                s_t = s_t1

                if tid == 0:
                    self.lock.acquire()
                    print '-' * 20, tid, '    ', self.t
                    print probs
                    print ' ' * 40, action_index
                    print 'r_t: ', r_t
                    self.lock.release()

                # for compute R_t
                s_t_last = s_t

                if terminal:
                    self.lock.acquire()
                    x_t = env.reset()
                    s_t = self.prepr.process(x_t)
                    self.lock.release()

                    ##terminal = False

            #self.lock.acquire()
            if self.tg_cfg['flag_stop_by_ctrl_c']:
                break
            #self.lock.release()

            if terminal:
                R_t = 0
            else:
                self.lock.acquire()
                # bootstrap from last state
                R_t = self.model.predict_v(s_t_last)[0][0]
                self.lock.release()

            R_batch = np.zeros(t)
            for i in reversed(range(t_start, t)):

                if term_batch[i]:
                    R_t = 0

                R_t = past_rewards[i] + self.gamma * R_t
                R_batch[i] = R_t

            # <3.1.1>
            #'''
            v_ts = []
            self.lock.acquire()
            for st in s_batch:
                #self.lock.acquire()
                #v = self.model.value_network.predict(st)[0][0]
                v = self.model.predict_v(st)[0][0]
                #self.lock.release()
                v_ts.append(v)
            #self.lock.release()
            #'''

            #self.lock.acquire()
            loss = self.train_a_batch(t, s_batch, a_batch, R_batch, v_ts)
            if tid == 0:
                print '=' * 20, tid, '    ', self.t
                print 'loss: ', loss
                print 'v: ', v_ts[-1]
                #print 'term_batch: ', term_batch
                #print 'past_rewards: ', past_rewards
                #print 'R_batch: ', R_batch
                #print 'v_ts: ', v_ts
            self.lock.release()

            if self.t > 0 and self.t % self.model_saved_per == 0:
                if self.model_saved_file_p:
                    self.lock.acquire()
                    self.model.save(self.model_saved_file_p,
                                    self.model_saved_file_v)
                    self.lock.release()

            self.t = self.t + 1


    def train_a_batch(self, tn, s_batch, a_batch, R_batch, v_ts):
        """
        """
        loss = 0.0

        p_targets = np.zeros((self.n_batch, self.n_actions))
        v_targets = np.array(R_batch)
        inputs = np.array(s_batch)
        inputs = inputs.reshape((inputs.shape[0], inputs.shape[2],
                                 inputs.shape[3], inputs.shape[4]))

        self.model.set_value(self.model.R_t, R_batch)
        self.model.set_value(self.model.a_t, a_batch)
        self.model.set_value(self.model.v_t, v_ts)

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
                self.lock.acquire()
                self.tg_cfg['flag_stop_by_ctrl_c'] = True
                self.lock.release()

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

                probs = self.model.predict_p(s_t)[0]
                action_index = self.sample_policy_action(self.n_actions, probs)

                #print s_t
                print probs
                print action_index

                x_t1_colored, r_t, terminal, info = self.env.step(action_index)
                s_t1 = self.prepr.process(x_t1_colored, s_t)

                s_t = s_t1
