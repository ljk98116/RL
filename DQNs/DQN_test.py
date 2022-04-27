import gym
from DQNs.DQN import DQN

env = gym.make('CartPole-v1')

def DQN_Test():
    dqn_agent = DQN(env=env)
    dqn_agent.fit()

if __name__ == "__main__":
    DQN_Test()