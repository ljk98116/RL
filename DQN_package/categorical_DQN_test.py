import gym
from categorical_DQN import categorical_DQN

env = gym.make('CartPole-v1')

def cDQN_Test():
    cdqn_agent = categorical_DQN(env=env)
    cdqn_agent.fit()

if __name__ == "__main__":
    cDQN_Test()