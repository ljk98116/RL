import gym
from n_step_DQN import n_step_DQN

env = gym.make('CartPole-v1')

def nDQN_Test():
    ndqn_agent = n_step_DQN(env=env)
    ndqn_agent.fit()

if __name__ == "__main__":
    nDQN_Test()