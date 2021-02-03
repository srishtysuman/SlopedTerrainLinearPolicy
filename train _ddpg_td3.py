import argparse
import random
import numpy as np
import torch

import gym
import matplotlib.pyplot as plt

from gym.envs.registration import registry, register, make, spec
from td3 import TD3
from ddpg import DDPG
#Custom environments that you want to use ----------------------------------------------------------------------------------------
register(id='Stoch2-v0',
           entry_point='gym_sloped_terrain.envs.stoch2_pybullet_env:Stoch2Env', 
           kwargs = {'gait' : "trot", 'render': False, 'action_dim': 20, 'stairs': 0} )
#---------------------------------------------------------------------------------------------------------------------------------

def rollout(agent, env, train=False, random=False):
    state = env.reset()
    episode_step, episode_return = 0, 0
    done = False
    while not done:
        if random:
            action = env.action_space.sample()
        else:
            action = agent.act(state, train=train)

        # print('agent_act')
        # print(agent.act)
        # temp = input()
        next_state, reward, done, info = env.step(action)
        # print('action = ', action,'action size = ',action.shape)
        # temp = input()
        # print('next_state = ', next_state,' size = ',type(next_state))
        # temp = input()
        # print('reward = ',reward,'reward type = ',type(reward))
        # temp = input()
        # print('info = ', info, 'type info = ', type(info))
        # temp = input()
        # print('done = ',done,'tpye done = ',type(done))
        # temp = input()

        episode_return += reward

        if train:
            not_done = 1.0 if (episode_step+1) == 10 else float(not done)
            agent.replay_buffer.append([state, action, [reward], next_state, [not_done]])
            agent._timestep += 1

        state = next_state
        episode_step += 1

    if train and not random:
        for _ in range(episode_step):
            agent.update_parameters()

    return episode_return

def evaluate(agent, env, n_episodes=10):
    returns = [rollout(agent, env, train=False, random=False) for _ in range(n_episodes)]
    # print('returns = ', returns)
    # temp = input()
    return np.mean(returns)

def train(agent, env, n_episodes=1000, n_random_episodes=10):
    for episode in range(n_episodes):
        train_return = rollout(agent, env, train=True, random=episode<n_random_episodes)
        print(f'Episode {episode}. Return {train_return}')
        episode_list.append(episode+1)
        reward_list.append(train_return)

        # if (episode+1) % 10 == 0:
        #     eval_return = evaluate(agent, env)
        #     print(f'Eval Reward {eval_return}')



if __name__ == '__main__':
    episode_list = []
    reward_list = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="td3")  # td3 or sac
    parser.add_argument("--env", default="pybullet") #pybullet


    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--latency", default=2, type=int)




    args = parser.parse_args()

    if args.env == 'pybullet':
        env = gym.make('Stoch2-v0')
    else:
        raise Exception('Unknown env')

    obs_size, act_size = env.observation_space.shape[0], env.action_space.shape[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env.seed(args.seed)
    env.action_space.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # if args.agent == 'td3':
    agent = TD3(device, obs_size, act_size)
    train(agent, env, n_episodes=100, n_random_episodes=10)
    plt.plot(episode_list,reward_list,label='td3')
    #plt.savefig('ddpg_stoch2.png')
    # elif args.agent == 'ddpg':
    np.save('td3_episode_list.npy',episode_list)
    np.save('td3_reward_list.npy',reward_list)
    
    episode_list=[]
    reward_list=[]
    agent = DDPG(device, obs_size, act_size)
    train(agent, env, n_episodes=100, n_random_episodes=10)
    plt.plot(episode_list,reward_list,label='ddpg')
    
    plt.legend()
    plt.savefig('ddpg_stoch2.png')
    
    np.save('ddpg_episode_list.npy',episode_list)
    np.save('ddpg_reward_list.npy',reward_list)
    
    # else:
        # raise Exception('Unknown agent')

    
