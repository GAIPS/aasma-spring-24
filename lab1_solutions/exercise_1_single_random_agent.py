import argparse

import numpy as np
from gym import Env

from aasma import Agent
from aasma.utils import compare_results
from aasma.wrappers import SingleAgentWrapper
from aasma.simplified_predator_prey import SimplifiedPredatorPrey


def run_single_agent(environment: Env, agent: Agent, n_episodes: int) -> np.ndarray:

    results = np.zeros(n_episodes)

    for episode in range(n_episodes):

        steps = 0
        terminal = False
        observation = environment.reset()
        while not terminal:
            steps += 1
            # TODO - Main Loop (4 lines of code)
            agent.see(observation)
            action = agent.action()
            next_obs, reward, terminal, info = environment.step(action)
            observation = next_obs

        environment.close()

        results[episode] = steps

    return results


class RandomAgent(Agent):

    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__("Random Agent")
        self.n_actions = n_actions

    def action(self) -> int:
        return np.random.randint(self.n_actions)

    def next(self, observation, action, next_observation, reward, terminal, info):
        # Not a learning agent
        pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    opt = parser.parse_args()

    # 1 - Setup environment
    environment = SimplifiedPredatorPrey(
        grid_shape=(7, 7),
        n_agents=1, n_preys=1,
        max_steps=100, required_captors=1
    )
    environment = SingleAgentWrapper(environment, agent_id=0)

    # 2 - Setup agent
    agent = RandomAgent(environment.action_space.n)

    # 3 - Evaluate agent
    results = {
        agent.name: run_single_agent(environment, agent, opt.episodes)
    }

    # 4 - Compare results
    compare_results(results, title="Random Agent on 'Predator Prey' Environment", colors=["orange"])
