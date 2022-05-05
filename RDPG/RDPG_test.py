from RDPG_Agent import RDPG_Agent
import gym

if __name__ == "__main__":
    env = gym.make('Pendulum-v1')
    rdpg_agent = RDPG_Agent(env)
    rdpg_agent.fit()
