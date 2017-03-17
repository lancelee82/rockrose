"""
"""

import random

import numpy as np

import argparse

import gym

from rockrose import preprocessor
from rockrose import replay_memory
from rockrose.models import unreal as rr_model_unr
from rockrose.trainers import unreal as rr_trainer_unr

from bluegym import env_bluelake


ENV_GAME_NAME = 'Flibird-v0'
TRAINER_THREAD_N = 10#4


def env_reg():
    env_bluelake.gym_env_register_bluelake(
        'gymbird', (288, 512),
        ENV_GAME_NAME,
        obs_type='image',
        frameskip=(1, 2)  # (1, 6)
    )


def env_make():
    env = gym.make(ENV_GAME_NAME)
    np.random.seed(123)
    env.seed(123)

    return env


class RRPreprImgGrayN_RZeroTo(preprocessor.RRPreprImgGrayN4R):
    def process_reward(self, r_t, *args, **kwargs):
        if r_t == 0.0:
            r_t = 0.1
        return r_t


class RRPreprImgGrayN_FailToLarge(preprocessor.RRPreprImgGrayN4R):
    def process_reward(self, r_t, *args, **kwargs):
        if r_t == -1.0:
            r_t = -10.0
        return r_t


def main():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m','--mode', help='train / play')
    parser.add_argument('-i','--model', help='model file names')
    args = parser.parse_args()

    env_reg()

    envs = []
    rmem = []
    for i in range(TRAINER_THREAD_N):
        envs.append(env_make())

        #rmm = replay_memory.ReplayMemory(50000)
        rmm = replay_memory.ReplayMemoryPrior(5000)
        rmem.append(rmm)

    #prepr = preprocessor.RRPreprImgGrayN(4, out_size=(84, 84))
    prepr = preprocessor.RRPreprImgGrayN4R(4, out_size=(84, 84))
    #prepr = RRPreprImgGrayN_RZeroTo(4, out_size=(84, 84))

    actn = len(envs[0]._action_set)
    md_cfg = {
        'input_shape': (4, 84, 84),
        #'input_shape': (4, 72, 72),
        'pc_wh': 84 / 4,
        'actn': actn,
        'lr': 1e-5,  #1e-4,  # 1e-6
        'pc_wh': 84 / 4,
        'pc_cw': 11,
        'pc_lambda': 1.0,
    }
    model = rr_model_unr.RRModelUnreal(md_cfg)

    trnr_cfg = {
        'thread_n': TRAINER_THREAD_N,
        #'if_render': True,
        'model_saved_file_p': 'models_saved/unr_flibird_1_p.h5',
        'model_saved_file_v': 'models_saved/unr_flibird_1_v.h5',
        'model_saved_per': 20,
        'use_rp': True,
        'use_vr': True,
        'use_pc': True,
        'pc_wh': 84 / 4,
    }

    if args.mode == 'train':
        trnr = rr_trainer_unr.RRTrainerUnreal(trnr_cfg, envs, model, prepr, rmem)
        #trnr.train_a_thread(0)
        trnr.train()
    else:
        trnr = rr_trainer_unr.RRTrainerUnreal(trnr_cfg, envs[0], model, prepr, rmem)
        trnr.play()


if __name__ == '__main__':
    main()
