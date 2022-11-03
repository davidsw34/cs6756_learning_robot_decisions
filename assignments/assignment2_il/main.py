import gym
from stable_baselines3.ppo import PPO
import torch.nn as nn
import argparse

from learners import *
from utils import *


def make_env():
    return gym.make("LunarLander-v2")


def get_expert():
    return PPO.load("./experts/LunarLander-v2/lunarlander_expert")


def get_expert_performance(env, expert):
    Js = []
    for _ in range(100):
        obs = env.reset()
        J = 0
        done = False
        hs = []
        while not done:
            action, _ = expert.predict(obs)
            obs, reward, done, info = env.step(action)
            hs.append(obs[1])
            J += reward
        Js.append(J)
    ll_expert_performance = np.mean(Js)
    return ll_expert_performance


def main(args):
    env = make_env()
    expert = get_expert()

    performance = get_expert_performance(env, expert)
    print('=' * 20)
    print(f'Expert performance: {performance}')
    print('=' * 20)

    # net + loss fn
    if args.truncate:
        net = create_net(input_dim=6, output_dim=4)
    else:
        net = create_net(input_dim=8, output_dim=4)

    loss_fn = nn.CrossEntropyLoss()

    if args.bc:
        # TODO: train BC
        # Things that need to be done:
        # - Roll out the expert for X number of trajectories (a standard amount is 10).
        # - Create our BC learner, and train BC on the collected trajectories.
        # - It's up to you how you want to structure your data!
        # - Evaluate the argmax_policy by printing the total rewards.

        # 1 expert trajectory
        bc = BC(net, loss_fn)
        policy = None
        epochs = 10
        for i in range(epochs):
            states, actions = expert_rollout(expert, env)
            policy = bc.learn(env=env, states=states, actions=actions)
        reward = eval_policy(policy, env, truncate=False)
        print(reward)

        # Code for Multiple Expert Rollouts
        # num_trajectories = 10
        # x = [expert_rollout(expert, env, truncate=False)
        #      for _ in range(num_trajectories)]
        # states = []
        # actions = []
        # # aggregate trajectories
        # for state_traj, action_traj in x:
        #     for state, action in zip(state_traj, action_traj):
        #         states.append(state)
        #         actions.append(action)
        # # perform normal BC
        # states = np.array(states)
        # actions = np.array(actions)
        # bc = BC(net, loss_fn)
        # policy = None
        # epochs = 10
        # for i in range(epochs):
        #     policy = bc.learn(env=env, states=states, actions=actions)
        # reward = eval_policy(policy, env, truncate=False)
        # print(reward)

    else:
        # TODO: train DAgger
        # Things that need to be done.
        # - Create our DAgger learner.
        # - Set up the training loop. Make sure it is fundamentally interactive!
        # - It's up to you how you want to structure your data!
        # - Evaluate the argmax_policy by printing the total rewards.
        dagger = DAgger(net, loss_fn, expert)
        policy = None
        epochs = 10
        for epoch in range(epochs):
            policy = dagger.learn(env, truncate=False)
        reward = eval_policy(policy, env, truncate=False)
        print(reward)


def get_args():
    parser = argparse.ArgumentParser(description='imitation')
    parser.add_argument('--bc', action='store_true',
                        help='whether to train BC or DAgger')
    parser.add_argument('--n_steps', type=int, default=10000,
                        help='number of steps to train learner')
    parser.add_argument('--truncate', action='store_true',
                        help='whether to truncate env')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(get_args())
