import gym
from Rainbow_DQN import Rainbow_DQN

env = gym.make('CartPole-v1')

def Rainbow_DQN_Test():
    rainbow_agent = Rainbow_DQN(env=env)
    rainbow_agent.fit()

if __name__ == "__main__":
    Rainbow_DQN_Test()