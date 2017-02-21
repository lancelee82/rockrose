"""
"""

#from __future__ import print_function

import sys
import random
import time

import threading

import numpy as np

import base as rr_trainer_base


class RRTrainerUnreal(rr_trainer_base.RRTrainerBase):
    def __init__(self, cfg, env, model, prepr, rmem, *args, **kwargs):
        super(RRTrainerUnreal, self).__init__(
            cfg, env, model, prepr, rmem, *args, **kwargs)

        self.use_pc = self.cfg.get('use_pc', False)
        self.use_vr = self.cfg.get('use_vr', False)
        self.use_rp = self.cfg.get('use_rp', False)

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

        self.pc_wh = self.cfg.get('pc_wh', 84 / 4)

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
                self.model.load(self.model_saved_file_p, self.model_saved_file_v)
            except Exception as e:
                print e

    def hook_env_render(self, env=None):
        if not env:
            env = self.env

        if self.if_render:
            env.render()

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
        rmem = self.rmem[tid]

        self.lock.acquire()
        x_t = env.reset()
        s_t = self.prepr.process(x_t)
        self.lock.release()

        terminal = False

        t_max = self.n_batch

        while self.t < self.n_train_steps:

            s_batch = []
            past_rewards = []
            a_batch = []

            # <3.1.2>
            v_ts = []

            term_batch = []

            last_a_rs = []

            last_action = 0
            last_reward = 0

            t = 0
            t_start = t

            #while not (terminal or ((t - t_start) == t_max)):
            while not ((t - t_start) == t_max):
                #self.lock.acquire()
                if self.tg_cfg['flag_stop_by_ctrl_c']:
                    break
                #self.lock.release()

                self.hook_env_render(env)

                self.lock.acquire()
                last_a_r = np.zeros((1, 1, self.n_actions + 1))
                last_a_r[0][0][last_action] = 1.0
                last_a_r[0][0][-1] = last_reward

                probs = self.model.predict_p([s_t, last_a_r])[0]
                v = self.model.predict_v([s_t, last_a_r])[0][0]
                self.lock.release()

                action_index = self.sample_policy_action(self.n_actions, probs)
                a_t = np.zeros([self.n_actions])
                a_t[action_index] = 1

                s_batch.append(s_t)
                a_batch.append(a_t)
                v_ts.append(v)
                last_a_rs.append(last_a_r)

                self.lock.acquire()
                x_t1_colored, r_t, terminal, info = env.step(action_index)
                s_t1 = self.prepr.process(x_t1_colored, s_t)
                self.lock.release()

                #r_t = np.clip(r_t, -1, 1)
                r_t = self.prepr.process_reward(r_t)
                past_rewards.append(r_t)

                term_batch.append(terminal)

                # save to replay memory
                rmem.append((s_t, action_index, r_t, s_t1, terminal, last_action, last_reward))

                t += 1

                s_t = s_t1

                last_action = action_index
                last_reward = r_t

                if tid == 0:
                    self.lock.acquire()
                    print '-' * 20, tid, '    ', self.t
                    print probs
                    print ' ' * 40, action_index
                    print 'r_t: ', r_t
                    self.lock.release()

                # for compute R_t
                s_t_last = s_t
                a_last = action_index
                r_last = r_t

                if terminal:
                    self.lock.acquire()
                    x_t = env.reset()
                    s_t = self.prepr.process(x_t)
                    self.lock.release()

                    #terminal = False

                    last_action = 0
                    last_reward = 0

            #self.lock.acquire()
            if self.tg_cfg['flag_stop_by_ctrl_c']:
                break
            #self.lock.release()

            if terminal:
                R_t = 0
            else:
                self.lock.acquire()
                last_a_r = np.zeros((1, 1, self.n_actions + 1))
                last_a_r[0][0][a_last] = 1.0
                last_a_r[0][0][-1] = r_last

                R_t = self.model.predict_v([s_t_last, last_a_r])[0][0] # bootstrap from last state
                self.lock.release()

            R_batch = np.zeros(t)
            for i in reversed(range(t_start, t)):

                if term_batch[i]:
                    R_t = 0

                R_t = past_rewards[i] + self.gamma * R_t
                R_batch[i] = R_t

            # <3.1.1>
            '''
            v_ts = []
            self.lock.acquire()
            for st in s_batch:
                #self.lock.acquire()
                #v = self.model.value_network.predict(st)[0][0]
                v = self.model.predict_v(st)[0][0]
                #self.lock.release()
                v_ts.append(v)
            self.lock.release()
            '''

            self.lock.acquire()
            loss = self.train_a_batch(t, s_batch, a_batch, R_batch, v_ts, last_a_rs)
            if tid == 0:
                print '=' * 20, tid, '    ', self.t
                print 'loss: ', loss
                print 'v: ', v_ts[-1]
                #print 'term_batch: ', term_batch
                #print 'past_rewards: ', past_rewards
                #print 'R_batch: ', R_batch
                #print 'v_ts: ', v_ts
            #self.lock.release()

            #self.lock.acquire()
            if self.use_rp:# and rmem.cnt > self.n_batch * 10:
                loss = self.train_a_batch_rp(t, rmem)
                if tid == 0:
                    print 'rp', '$' * 60
                    print 'rp loss: ', loss
            #self.lock.release()

            #self.lock.acquire()
            if self.use_vr:# and rmem.cnt > self.n_batch * 10:
                loss = self.train_a_batch_vr(t, rmem)
                if tid == 0:
                    print 'vr', '%' * 60
                    print 'vr loss: ', loss
            #self.lock.release()

            #self.lock.acquire()
            if self.use_pc:# and rmem.cnt > self.n_batch * 10:
                loss = self.train_a_batch_pc(t, rmem)
                if tid == 0:
                    print 'pc', '^' * 60
                    print 'pc loss: ', loss
            self.lock.release()

            if self.t > 0 and self.t % self.model_saved_per == 0:
                if self.model_saved_file_p:
                    self.lock.acquire()
                    self.model.save(self.model_saved_file_p, self.model_saved_file_v)
                    self.lock.release()

            self.t = self.t + 1


    def train_a_batch(self, tn, s_batch, a_batch, R_batch, v_ts, last_a_rs):
        """
        """
        loss = 0.0

        p_targets = np.zeros((self.n_batch, self.n_actions))
        v_targets = np.array(R_batch)
        inputs = np.array(s_batch)
        inputs = inputs.reshape((inputs.shape[0], inputs.shape[2], inputs.shape[3], inputs.shape[4]))
        last_ars = np.array(last_a_rs)
        last_ars = last_ars.reshape((last_ars.shape[0], last_ars.shape[2], last_ars.shape[3]))

        self.model.set_value(self.model.R_t, R_batch)
        self.model.set_value(self.model.a_t, a_batch)
        self.model.set_value(self.model.v_t, v_ts)

        loss += self.model.train_on_batch_p([inputs, last_ars], p_targets)
        loss += self.model.train_on_batch_v([inputs, last_ars], v_targets)
        return loss


    def train_a_batch_rp(self, t, rmem):
        minibatch = rmem.sample_zr(4)
        if not minibatch:
            print 'rp', '#' * 100
            return

        inputs = []
        targets = []

        '''
        for i in range(3):
            inputs.append(minibatch[i][0])

        inputs = np.array(inputs)
        inputs = inputs.reshape((inputs.shape[0], inputs.shape[2], inputs.shape[3], inputs.shape[4]))
        '''
        #x#inputs = np.array(minibatch[3][0][:, :4 - 1, :, :])  # the one s_t has already 4 frames !!!
        inputs = np.array(minibatch[3][0])

        r = minibatch[3][2]
        rp = [0.0, 0.0, 0.0]
        if r == 0.0:
            rp[0] = 1.0
        elif r > 0.0:
            rp[1] = 1.0
        else:
            rp[2] = 1.0
        targets.append(rp)
        targets = np.array(targets)

        loss = self.model.train_on_batch_rp(inputs, targets)
        return loss


    def train_a_batch_vr(self, t, rmem):
        minibatch = rmem.sample_sq(self.n_batch)
        if not minibatch:
            print 'vr', '#' * 100
            return

        s_t = minibatch[-1][0]
        terminal = minibatch[-1][4]
        a_last = minibatch[-1][5]
        r_last = minibatch[-1][6]

        s_batch = np.zeros((self.n_batch, s_t.shape[1], s_t.shape[2], s_t.shape[3]))
        R_batch = np.zeros(self.n_batch)
        last_a_rs = np.zeros((self.n_batch, 1, 1, self.n_actions + 1))

        if terminal:
            R_t = 0
        else:
            #self.lock.acquire()
            last_a_r = np.zeros((1, 1, self.n_actions + 1))
            last_a_r[0][0][a_last] = 1.0
            last_a_r[0][0][-1] = r_last
            R_t = self.model.predict_v([s_t, last_a_r])[0][0] # bootstrap from last state
            #self.lock.release()

        for i in reversed(range(0, len(minibatch))):
            state_t = minibatch[i][0]
            action_t = minibatch[i][1]  # this is action index
            reward_t = minibatch[i][2]
            state_t1 = minibatch[i][3]
            terminal = minibatch[i][4]
            a_last = minibatch[i][5]
            r_last = minibatch[i][6]

            s_batch[i:i + 1] = state_t

            if terminal:
                R_t = 0

            R_t = reward_t + self.gamma * R_t
            R_batch[i] = R_t

            last_a_r = np.zeros((1, 1, self.n_actions + 1))
            last_a_r[0][0][a_last] = 1.0
            last_a_r[0][0][-1] = r_last
            last_a_rs[i:i + 1] = last_a_r

        v_targets = np.array(R_batch)
        inputs = np.array(s_batch)
        last_ars = np.array(last_a_rs)
        last_ars = last_ars.reshape((last_ars.shape[0], last_ars.shape[2], last_ars.shape[3]))

        loss = self.model.train_on_batch_vr([inputs, last_ars], v_targets)
        return loss


    def train_a_batch_pc(self, t, rmem):
        minibatch = rmem.sample_sq(self.n_batch)
        if not minibatch:
            print 'pc', '#' * 100
            return

        s_t = minibatch[-1][0]
        terminal = minibatch[-1][4]
        a_last = minibatch[-1][5]
        r_last = minibatch[-1][6]

        s_batch = np.zeros((self.n_batch, s_t.shape[1], s_t.shape[2], s_t.shape[3]))
        pc_a = np.zeros((self.n_batch, self.n_actions))
        pc_r = np.zeros((self.n_batch, self.pc_wh, self.pc_wh))
        last_a_rs = np.zeros((self.n_batch, 1, 1, self.n_actions + 1))

        if terminal:
            pc_R_t = np.zeros([self.pc_wh, self.pc_wh], dtype=np.float32)
        else:
            #self.lock.acquire()
            last_a_r = np.zeros((1, 1, self.n_actions + 1))
            last_a_r[0][0][a_last] = 1.0
            last_a_r[0][0][-1] = r_last
            last_a_r = np.array(last_a_r, dtype=np.float32)

            st_b = np.repeat(s_t, self.n_batch, axis=0)
            ar_b = np.repeat(last_a_r, self.n_batch, axis=0)
            pc_R_t = self.model.predict_pc_q_max([st_b, ar_b])[0] # bootstrap from last state
            #self.lock.release()

        for i in reversed(range(0, len(minibatch))):
            state_t = minibatch[i][0]
            action_t = minibatch[i][1]  # this is action index
            reward_t = minibatch[i][2]
            state_t1 = minibatch[i][3]
            terminal = minibatch[i][4]
            a_last = minibatch[i][5]
            r_last = minibatch[i][6]

            s_batch[i:i + 1] = state_t

            a_t = np.zeros([self.n_actions])
            a_t[action_t] = 1
            pc_a[i:i + 1] = a_t

            if terminal:
                pc_R_t = np.zeros([self.pc_wh, self.pc_wh], dtype=np.float32)

            pixel_change = self._calc_pixel_change(state_t1[-1], state_t[-1])
            pc_R_t = pixel_change + self.gamma * pc_R_t
            pc_r[i:i + 1] = pc_R_t

            last_a_r = np.zeros((1, 1, self.n_actions + 1))
            last_a_r[0][0][a_last] = 1.0
            last_a_r[0][0][-1] = r_last
            last_a_rs[i:i + 1] = last_a_r

        targets = np.zeros([self.n_batch, self.n_actions, self.pc_wh, self.pc_wh], dtype=np.float32)
        inputs = np.array(s_batch)
        last_ars = np.array(last_a_rs)
        last_ars = last_ars.reshape((last_ars.shape[0], last_ars.shape[2], last_ars.shape[3]))

        pc_a = np.array(pc_a)
        pc_r = np.array(pc_r)

        self.model.set_value(self.model.pc_a, pc_a)
        self.model.set_value(self.model.pc_r, pc_r)

        loss = self.model.train_on_batch_pc_q([inputs, last_ars], targets)
        return loss


    def _subsample(self, a, average_width):
        s = a.shape
        sh = s[0]//average_width, average_width, s[1]//average_width, average_width
        return a.reshape(sh).mean(-1).mean(1)  


    def _calc_pixel_change(self, state, last_state):
        d = np.absolute(state - last_state)
        m = np.mean(d, 2)
        c = self._subsample(m, 4)
        return c


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

            last_action = 0
            last_reward = 0.0

            while not terminal:

                self.hook_env_render()

                last_a_r = np.zeros((1, 1, self.n_actions + 1))
                last_a_r[0][0][last_action] = 1.0
                last_a_r[0][0][-1] = last_reward

                probs = self.model.predict_p([s_t, last_a_r])[0]
                action_index = self.sample_policy_action(self.n_actions, probs)

                #print s_t
                print probs
                print action_index

                x_t1_colored, r_t, terminal, info = self.env.step(action_index)
                s_t1 = self.prepr.process(x_t1_colored, s_t)

                s_t = s_t1

                last_action = action_index
                last_reward = r_t
