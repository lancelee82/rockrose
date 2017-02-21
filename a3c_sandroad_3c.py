"""
"""

import random

import numpy as np

import gym

from rockrose import preprocessor
from rockrose import replay_memory
from rockrose.models import a3cc as rr_model_a3cc
from rockrose.trainers import a3cc as rr_trainer_a3cc

from bluegym import env_bluelake


ENV_GAME_NAME = 'Sandroad-v0'
TRAINER_THREAD_N = 16#10#4


def env_reg():
    env_bluelake.gym_env_register_bluelake(
        'gymroad', (640, 480),
        ENV_GAME_NAME,
        obs_type='image',
        frameskip=(1, 2)
        #frameskip=(1, 3)
        #frameskip=(1, 5)
    )


def env_make():
    env = gym.make(ENV_GAME_NAME)
    np.random.seed(123)
    env.seed(123)

    return env


class RRPreprImgGrayN_FailToLarge(preprocessor.RRPreprImgGrayN4R):
    def process_reward(self, r_t, *args, **kwargs):
        if r_t == -1.0:
            r_t = -10.0
        return r_t


def main():
    env_reg()

    envs = []
    for i in range(TRAINER_THREAD_N):
        envs.append(env_make())

    rmem = replay_memory.ReplayMemory(50000)

    #t#prepr = preprocessor.RRPreprImgGrayN(4, out_size=(84, 84))
    prepr = preprocessor.RRPreprImgGrayN4R(4, out_size=(84, 84))
    #prepr = RRPreprImgGrayN_FailToLarge(4, out_size=(84, 84))

    actn = len(envs[0]._action_set)
    md_cfg = {
        'input_shape': (4, 84, 84),
        'actn': actn,
        'lr': 1e-5,  #5e-5,  # 1e-3,  #1e-4,  # 1e-6
    }
    model = rr_model_a3cc.RRModelA3CConvPV(md_cfg)

    trnr_cfg = {
        'thread_n': TRAINER_THREAD_N,
        #'if_render': True,
        'model_saved_file_p': 'models_saved/a3c_sandroad_3c_1_p.h5',  # reward_24
        'model_saved_file_v': 'models_saved/a3c_sandroad_3c_1_v.h5',
        #'model_saved_file_p': 'models_saved/a3c_sandroad_3c_2_p.h5',  # reward_25
        #'model_saved_file_v': 'models_saved/a3c_sandroad_3c_2_v.h5',
        'model_saved_per': 100,
    }
    if 1:
        trnr = rr_trainer_a3cc.RRTrainerA3C(trnr_cfg, envs, model, prepr, rmem)
        #trnr.train_a_thread(0)
        trnr.train()
    else:
        trnr = rr_trainer_a3cc.RRTrainerA3C(trnr_cfg, envs[0], model, prepr, rmem)
        trnr.play()


if __name__ == '__main__':
    main()
