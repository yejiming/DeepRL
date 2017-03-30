#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import csv

import numpy as np
import tensorflow as tf
from DeepRL.hw1 import tf_util

import gym
from DeepRL.hw1 import load_policy
from DeepRL.hw1.networks import solutions
from DeepRL.hw1.conf import constants


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    expert_data = get_expert_data(args)
    solution = train(expert_data)
    max_steps, mean, std, observations = test(args, solution)
    write_result(args, max_steps, mean, args.envname+"_results.csv")

    dagger_data = dagger(args, observations)
    total_data = {
        "observations": np.vstack((expert_data["observations"], dagger_data["observations"])),
        "actions": np.vstack((expert_data["actions"], dagger_data["actions"]))
    }
    tf.reset_default_graph()
    solution = train(total_data)
    max_steps, mean, std, _ = test(args, solution)
    write_result(args, max_steps, mean, args.envname+"_dagger_results.csv")


def write_result(args, max_steps, mean, filename, is_dagger=False):
    with open("logs/"+filename, "a") as f:
        writer = csv.writer(f)
        if is_dagger:
            row = [args.num_rollouts, max_steps, constants.N_EPOCHS,
                   constants.BATCH_SIZE, constants.LEARNING_RATE,
                   mean / max_steps * 100, constants.DAGGER_ITER]
        else:
            row = [args.num_rollouts, max_steps, constants.N_EPOCHS, constants.BATCH_SIZE,
                   constants.LEARNING_RATE, mean / max_steps * 100]
        while True:
            try:
                writer.writerow(row)
                break
            except IOError:
                pass


def get_expert_data(args):
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None, :])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

    expert_data = {'observations': np.array(observations),
                   'actions': np.array(actions)}
    action_shape = expert_data["actions"].shape
    expert_data["actions"] = expert_data["actions"].reshape(action_shape[0], action_shape[2])
    return expert_data


def train(data):
    num_features = data["observations"].shape[1]
    num_outputs = data["actions"].shape[1]
    solution = solutions.SimpleSolution(num_features, num_outputs)
    solution.fit(data)
    return solution


def dagger(args, observations):
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    actions = []

    with tf.Session():
        tf_util.initialize()
        for obs in observations:
            action = policy_fn(obs[None, :])
            actions.append(action)

    dagger_data = {"observations": np.array(observations),
                   "actions": np.array(actions)}
    action_shape = dagger_data["actions"].shape
    dagger_data["actions"] = dagger_data["actions"].reshape(action_shape[0], action_shape[2])
    return dagger_data


def test(args, solution):
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    for i in range(constants.DAGGER_ITER):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = solution.predict(obs)
            observations.append(obs)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))
    observations = np.array(observations)

    return max_steps, np.mean(returns), np.std(returns), observations


if __name__ == '__main__':
    main()
