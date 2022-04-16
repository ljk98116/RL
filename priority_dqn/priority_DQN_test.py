import gym
from priority_DQN import priority_DQN

env = gym.make('CartPole-v1')

def priority_DQN_Test():
    pdqn_agent = priority_DQN(env=env)
    pdqn_agent.fit()

if __name__ == "__main__":
    priority_DQN_Test()