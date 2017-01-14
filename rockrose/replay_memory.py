"""
"""

import os.path
import random
import time

from collections import namedtuple, deque, OrderedDict
import cPickle as pickle


class ReplayMemory(object):
    def __init__(self, mem_max=10000):
        self.mem_max = mem_max

        self.dq = deque()
        self.cnt = 0

    def append(self, item):
        self.cnt += 1
        self.dq.append(item)
        if len(self.dq) > self.mem_max:
            self.dq.popleft()

    def sample(self, cnt):
        s = random.sample(self.dq, cnt)
        return s

    def sample_sq(self, cnt):
        s = []
        if len(self.dq) < cnt + 1:
            return s
        idx = random.randint(0, len(self.dq) - cnt - 1)
        for i in range(cnt):
            s.append(self.dq[idx + i])
        return s

    def is_full(self):
        return len(self.dq) >= self.mem_max


class ReplayMemoryArchive(ReplayMemory):
    def __init__(self, mem_max=10000, pth='', sv_per=1000):
        super(ReplayMemoryArchive, self).__init__(mem_max)

        self.pth = pth
        self.sv_per = sv_per

        self.sv_idx = 0
        self.sv_cnt = 0

    def save(self, to_f=None):
        d = deque()
        if self.cnt <= self.mem_max:
            for i in range(self.sv_per):
                d.append(self.dq[self.sv_idx * self.sv_per + i])
        else:
            for i in range(self.sv_per):
                d.append(self.dq[self.mem_max - self.sv_per + i])

        if not to_f:
            to_f = self.sv_path_by_time()

        with open(to_f, 'wb') as fo:
            pickle.dump(d, fo, protocol=2)

            self.sv_idx += 1
            self.sv_cnt = 0

    def sv_path_by_time(self):
        return self.pth + str(int(time.time())) + '.pkl'

    def load(self, fi_pkl):
        with open(fi_pkl, 'rb') as fi:
            self.dq = pickle.load(fi)

    # override
    def append(self, item):
        super(ReplayMemoryArchive, self).append(item)

        self.sv_cnt += 1
        if self.sv_cnt >= self.sv_per:
            self.save()

    def _encode(self, item):
        return pickle.dumps(item, protocol=2)

    def _decode(self, encoded_item):
        return pickle.loads(encoded_item)



class ReplayMemoryPrior(ReplayMemory):
    def __init__(self, mem_max=10000):
        super(ReplayMemoryPrior, self).__init__(mem_max)

        self.dq_zero = deque()  # = 0
        self.dq_non_zero = deque()  # <> 0

        self.top = 0

    # override
    def append(self, item, *args, **kwargs):
        super(ReplayMemoryPrior, self).append(item)

        rw = kwargs.get('reward', 0.0)
        if rw == 0.0:
            self.dq_zero.append(self.cnt - 1)
        else:
            self.dq_non_zero.append(self.cnt - 1)

        if self.is_full():
            self.top += 1

        if len(self.dq_zero) > 0:
            if self.dq_zero[0] < self.top:
                self.dq_zero.popleft()

        if len(self.dq_non_zero) > 0:
            if self.dq_non_zero[0] < self.top:
                self.dq_non_zero.popleft()


    def sample_zr(self, cnt):

        rnd = random.random()
        if rnd < 0.5:
            from_zero = True
        else:
            from_zero = False

        if len(self.dq_zero) == 0:
            from_zero = False

        if len(self.dq_non_zero) == 0:
            from_zero = True

        if from_zero:
            idx = random.choice(self.dq_zero)
        else:
            idx = random.choice(self.dq_non_zero)

        idx -= self.top

        if idx < cnt - 1:
            return None

        s = []
        for i in range(cnt):
            s.append(self.dq[idx - cnt + 1 + i])

        return s
