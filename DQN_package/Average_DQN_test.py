import gym
from Average_DQN import Average_DQN

env = gym.make('CartPole-v1')

def Average_DQN_Test():
    adqn_agent = Average_DQN(env=env)
    adqn_agent.fit()

if __name__ == "__main__":
    Average_DQN_Test()