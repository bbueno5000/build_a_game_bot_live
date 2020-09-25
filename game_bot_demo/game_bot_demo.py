"""
DOCSTRING
"""
import gym
import numpy

class Basic:
    """
    DOCSTRING
    """
    def __call__(self):
        env = gym.make('CartPole-v0')
        env.reset()
        for _ in range(1000):
            env.render()
            env.step(env.action_space.sample())

class HillClimbing:
    """
    DOCSTRING
    """
    def run_episode(self, env, parameters):
        """
        DOCSTRING
        """
        observation = env.reset()
        totalreward = 0
        for _ in range(200):
            env.render()
            action = 0 if numpy.matmul(parameters, observation) < 0 else 1
            observation, reward, done, info = env.step(action)
            totalreward += reward
            if done:
                break
        return totalreward

    def train(self, submit):
        """
        hill climbing algo training
        """
        env = gym.make('CartPole-v0')
        episodes_per_update = 5
        noise_scaling = 0.1
        parameters = numpy.random.rand(4) * 2 -1
        bestreward = 0
        for _ in range(2000):
            newparams = parameters + (numpy.random.rand(4) * 2 - 1) * noise_scaling
            reward = run_episode(env, newparams)
            print("reward %d best %d" % (reward, bestreward))
            if reward > bestreward: 
                bestreward = reward
                parameters = newparams
                if reward == 200:
                    break

if __name__ == '__main__':
    r = HillClimbing.train(submit=False)
    print(r)
