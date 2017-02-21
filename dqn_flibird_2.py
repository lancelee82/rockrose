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


class RRPreprImgGrayN_RZeroTo(preprocessor.RRPreprImgGrayN4R):
    def process_reward(self, r_t, *args, **kwargs):
        if r_t == 0.0:
            r_t = 0.1
        return r_t


def env_reg():

    args_env_name = 'Gymbird-v0'
    env_bluelake.gym_env_register_bluelake(
        'gymbird', (288, 512),
        args_env_name,
        obs_type='image',
        frameskip=(1, 2))

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

    #prepr = preprocessor.RRPreprImgGrayN4R(4, out_size=(84, 84))
    prepr = RRPreprImgGrayN_RZeroTo(4, out_size=(84, 84))

    actn = len(env._action_set)
    md_cfg = {
        'input_shape': (4, 84, 84),
        'actn': actn,
    }
    model = rr_model_dqn.RRModelDQNConv(md_cfg)

    if args.model:
        model_saved_file = args.model
    else:
        model_saved_file = 'models_saved/dqn_flibird_2_1.h5'

    if args.mode == 'play':
        trnr_cfg_play = {
            'is_play': True,
            'init_epsilon': 0.0001,
            #'if_render': True,
            'model_saved_file': model_saved_file,
        }
        trnr = rr_trainer_dqn.RRTrainerDQN(trnr_cfg_play, env, model, prepr, rmem)
        trnr.play()
    else:
        trnr_cfg_train = {
            'is_play': False,
            'init_epsilon': 0.3,
            #'if_render': True,
            'model_saved_file': model_saved_file,
            'model_saved_per': 300,
        }
        trnr = rr_trainer_dqn.RRTrainerDQN(trnr_cfg_train, env, model, prepr, rmem)
        trnr.train()


if __name__ == '__main__':
    main()
