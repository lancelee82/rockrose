"""
"""

import random

import numpy as np

import argparse

import gym

from rockrose import preprocessor
from rockrose import replay_memory
from rockrose.models import dqn as rr_model_dqn
from rockrose.trainers import dqn as rr_trainer_dqn

from bluegym import env_bluelake


FT_BLK_N = 4#10
FT_BLK_CN = 3#6
FT_MAX_DO_CNT_FCT = 2#3#6#4


def env_reg():

    args_env_name = 'Gymfdt-v0'
    env_bluelake.gym_env_register_bluelake(
        'gymfdt', (800, 550),
        args_env_name,
        obs_type='image',
        frameskip=(1, 2),  # (1, 6)
        # floodit spec kwargs:
        blk_n=FT_BLK_N,
        blk_cn=FT_BLK_CN,
        max_do_cnt=FT_BLK_N * FT_MAX_DO_CNT_FCT)

    env = gym.make(args_env_name)
    np.random.seed(123)
    env.seed(123)

    return env


def main():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m','--mode', help='train / play')
    parser.add_argument('-i','--model', help='model file names')
    args = parser.parse_args()

    env = env_reg()

    rmem = replay_memory.ReplayMemory(50000)

    prepr = preprocessor.RRPreprImgRGB()

    actn = len(env._action_set)
    md_cfg = {
        'input_shape': (3, 84, 84),
        'actn': actn,
    }
    model = rr_model_dqn.RRModelDQNConv(md_cfg)

    if args.mode == 'train':
        trnr_cfg_train = {
            'is_play': False,
            'init_epsilon': 0.3,
            #'if_render': True,
            'model_saved_file': 'models_saved/dqn_floodit_1_1.h5',
            'model_saved_per': 300,
        }
        trnr = rr_trainer_dqn.RRTrainerDQN(trnr_cfg_train, env, model, prepr, rmem)
        trnr.train()
    else:
        trnr_cfg = {
            'is_play': True,
            'init_epsilon': 0.0001,
            'model_saved_file': 'models_saved/dqn_floodit_1_1.h5',
        }
        trnr = rr_trainer_dqn.RRTrainerDQN(trnr_cfg, env, model, prepr, rmem)
        trnr.play()


if __name__ == '__main__':
    main()
