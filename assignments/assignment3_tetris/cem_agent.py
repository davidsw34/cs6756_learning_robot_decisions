from copy import deepcopy
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from gym_simplifiedtetris.agents.base import BaseAgent
from gym_simplifiedtetris.envs._simplified_tetris_engine import SimplifiedTetrisEngine
from gym_simplifiedtetris.envs import SimplifiedTetrisBinaryEnv as Tetris
from tqdm import trange

from gym_simplifiedtetris.agents.dellacherie import DellacherieAgent
import pickle


class CEMAgent(DellacherieAgent):
    def __init__(self):
        self.weights = np.random.rand((6)).astype("double")

    def learn(self, epochs, eval, mean, variance, num_vecs, rho, Z_t, val):
        # initial normal distribution
        mean_vec = mean * np.ones(len(self.weights))
        variance_vec = variance * np.ones(len(self.weights))

        best_vecs = self.weights
        for epoch in range(epochs):
            vec_list = []
            # sample distribution to get weight vectors and scores
            for _ in range(num_vecs):
                vec = np.random.normal(mean_vec, np.sqrt(variance_vec))
                score = eval(vec)
                vec_list.append((vec, score))
            vec_list = sorted(vec_list, key=lambda tup: tup[1], reverse=True)

            # save best vectors
            keep = int(rho * num_vecs)
            best = vec_list[:keep]
            # don't consider vectors that score 0 points
            if not best[0][1] == 0:
                index = 0
                for i in range(index+1, len(best)):
                    if best[i][1] == 0:
                        best = best[:index+1]
                        break
                    else:
                        index = i + 1
                best_vecs = [tup[0].astype("double") for tup in best]
                mean_vec = np.mean(best_vecs, axis=0).astype("double")

                # remove nans
                for index in range(len(mean_vec)):
                    if np.isnan(mean_vec[index]):
                        mean_vec[index] = 0

                variance_vec = (np.var(best_vecs, axis=0) +
                                Z_t).astype("double")
                for index in range(len(variance_vec)):
                    if np.isnan(variance_vec[index]):
                        variance_vec[index] = 1
            # print(variance_vec)
            # print(mean_vec)
            # print(best)

            # evaluate after each epoch
            # score = val(mean_vec)
            # print("Epoch " + str(epoch) + " scored " + str(score) + " points")

        self.weights = np.array(mean_vec).astype("double")
        return np.array(self.weights)

    def predict(self, env: SimplifiedTetrisEngine, weights, **kwargs: Any) -> int:
        """Return the action yielding the largest heuristic score.

        Ties are separated using a priority rating, which is based on the translation and rotation of the current piece.

        :param env: environment that the agent resides in.
        :return: action with the largest rating (where ties are separated based on the priority).
        """
        scores = self.get_score(env, weights)
        return np.argmax(scores)

    def play_game(self, env, weights, max_steps=10000):
        obs = env.reset()
        score = 0
        for step in range(0, max_steps):
            action = self.predict(env, weights)
            obs, reward, done, info = env.step(action)
            score += reward
            if done:
                break
        return score

    def get_score(self, env: SimplifiedTetrisEngine, weights) -> np.ndarray:
        """Compute and return the Dellacherie feature set values.

        :param env: environment that the agent resides in.
        :return: Dellacherie feature values.
        """
        dell_scores = np.empty((env.num_actions,), dtype="double")

        available_actions = env._engine._all_available_actions[env._engine._piece._id]

        for action, (translation, rotation) in available_actions.items():
            old_grid = deepcopy(env._engine._grid)
            old_colour_grid = deepcopy(env._engine._colour_grid)
            old_anchor = deepcopy(env._engine._anchor)

            env._engine._rotate_piece(rotation)
            env._engine._anchor = [translation, 0]

            env._engine._hard_drop()
            env._engine._update_grid(True)
            env._engine._clear_rows()

            feature_values = np.array(
                [func(env) for func in self._get_dell_funcs()], dtype="double"
            )
            dell_scores[action] = np.dot(feature_values, weights)

            env._engine._update_grid(False)

            env._engine._grid = deepcopy(old_grid)
            env._engine._colour_grid = deepcopy(old_colour_grid)
            env._engine._anchor = deepcopy(old_anchor)

        best_actions = np.argwhere(
            dell_scores == np.amax(dell_scores)).flatten()
        is_a_tie = len(best_actions) > 1

        # Resort to the priorities if there is a tie.
        if is_a_tie:
            return self._get_priorities(
                best_actions=best_actions,
                available_actions=available_actions,
                x_spawn_pos=env._width_ / 2 + 1,
                num_actions=env.num_actions,
            )
        return dell_scores

    def save(self):
        with open('weights.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(self.weights, f)

    def load(self):
        with open('weights.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
            self.set_weights(pickle.load(f))

    def set_weights(self, weights):
        self.weights = weights


if __name__ == "__main__":
    env = Tetris(grid_dims=(20, 10), piece_size=4)
    obs = env.reset()

    agent = CEMAgent()

    def eval(weights): return agent.play_game(env, weights, max_steps=100)

    def val(weights): return agent.play_game(env, weights, max_steps=10000)

    agent.learn(epochs=5, eval=eval, mean=0, variance=100,
                num_vecs=100, rho=.1, Z_t=4, val=val)

    agent.save()

    print(agent.weights)

    obs = env.reset()
    print(agent.play_game(env, agent.weights, max_steps=10000))
