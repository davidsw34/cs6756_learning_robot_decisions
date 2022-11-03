from utils import *
from torch import optim
import numpy as np

'''Learner file (BC + DAgger)'''


class BC:
    def __init__(self, net, loss_fn):
        self.net = net
        self.loss_fn = loss_fn

        self.opt = optim.Adam(self.net.parameters(), lr=3e-4)

    def learn(self, env, states, actions, n_steps=1e4, truncate=True):
        # TODO: Implement this method. Return the final greedy policy (argmax_policy).
        for i in range(int(n_steps)):
            index = np.random.randint(
                0, len(states), size=32)  # get batch of 32
            self.opt.zero_grad()
            output = self.net(torch.from_numpy(
                states[index]).to(torch.float32))
            loss = self.loss_fn(
                output, torch.flatten(torch.from_numpy(np.array(actions)[index])))
            loss.backward()
            self.opt.step()

        print("Loss: " + str(loss.item()))
        policy = argmax_policy(self.net)
        print("Reward: " + str(eval_policy(policy, env, truncate=False)))
        return policy


class DAgger:
    def __init__(self, net, loss_fn, expert):
        self.net = net
        self.loss_fn = loss_fn
        self.expert = expert

        self.opt = optim.Adam(self.net.parameters(), lr=3e-4)

    def learn(self, env, n_steps=1e4, truncate=True):
        # TODO: Implement this method. Return the final greedy policy (argmax_policy).
        # Make sure you are making the learning process fundamentally expert-interactive.

        num_rollouts = 10

        for r in range(num_rollouts):  # 10 rollouts
            if r == 0:  # first rollout, use expert to get better results
                x = [expert_rollout(self.expert, env, truncate=truncate)
                     for _ in range(10)]
            else:  # otherwise use network
                x = [rollout(self.net, env, truncate=truncate)
                     for _ in range(10)]  # number trajectories
            states = []
            actions = []
            # aggregate trajectories
            for state_traj, action_traj in x:
                for state, action in zip(state_traj, action_traj):
                    states.append(state)
                    if r == 0:
                        actions.append(action)
                    else:
                        actions.append(
                            np.argmax(expert_policy(self.expert, state)))
            # perform normal BC
            states = np.array(states)
            actions = np.array(actions)
            for i in range(int(n_steps/num_rollouts)):
                index = np.random.randint(
                    0, len(states), size=32)  # get batch of 32
                self.opt.zero_grad()
                output = self.net(torch.from_numpy(
                    states[index]).to(torch.float32))
                loss = self.loss_fn(
                    output, torch.flatten(torch.from_numpy(np.array(actions)[index])))
                loss.backward()
                self.opt.step()
        print("Loss: " + str(loss.item()))
        policy = argmax_policy(self.net)
        print("Reward: " + str(eval_policy(policy, env, truncate=False)))
        return policy
