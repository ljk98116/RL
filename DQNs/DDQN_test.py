import gym
from DQNs.DDQN import DDQN

env = gym.make('CartPole-v1')

def DDQN_Test():
    ddqn_agent = DDQN(env=env)
    ddqn_agent.fit()

if __name__ == "__main__":
    DDQN_Test()