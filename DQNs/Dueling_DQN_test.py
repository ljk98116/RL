import gym
from DQNs.Dueling_DQN import Dueling_DQN

env = gym.make('CartPole-v1')

def Dueling_DQN_Test():
    ddqn_agent = Dueling_DQN(env=env)
    ddqn_agent.fit()

if __name__ == "__main__":
    Dueling_DQN_Test()