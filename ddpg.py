import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import random
from collections import deque



class MLP(nn.Module):
    """ MLP with dense connections """
    def __init__(self, input_size, output_size, hidden_size, num_hidden_layers=3):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        hidden_size_aug = hidden_size + input_size
        self.linear_in = nn.Linear(input_size, hidden_size)
        hidden_layers = []
        for i in range(self.num_hidden_layers):
            hidden_layers.append(nn.Linear(hidden_size_aug, hidden_size))
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.linear_out = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        x = F.relu(self.linear_in(inp))
        for i in range(self.num_hidden_layers):
            x = torch.cat([x, inp], dim=1)
            x = F.relu(self.hidden_layers[i](x))
        return self.linear_out(x)

## Critic (value) network maps state, action pair to value
class Critic(nn.Module):
    """ Single Q-networks """
    def __init__(self, obs_size, act_size, hidden_size):
        super().__init__()
        self.net = MLP(obs_size+act_size, 1, hidden_size)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        return self.net(state_action)

## Actor directly maps states to action
class Actor(nn.Module):
    def __init__(self, obs_size, act_size, hidden_size, max_action):
        super().__init__()
        self.net = MLP(obs_size, act_size, hidden_size)
        self.max_action = max_action

    def forward(self, state):
        x = self.net(state)
        action = torch.tanh(x) * self.max_action
        return action

    def act(self, state, device, noise=0):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        action = self.forward(state)
        return action[0].detach().cpu().numpy()


class DDPG:
    def __init__(self,device,obs_size,act_size,max_action=1,hidden_size=256,gamma=0.99,tau=0.005,policy_noise=0.2,
    									noise_clip=0.5,policy_freq=1,exploration_noise=0.1):
        self.device = device 
        self.act_size = act_size
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.exploration_noise = exploration_noise
        self._timestep = 0
	

	## Randomly initialize critic network
        self.critic = Critic(obs_size, act_size, hidden_size).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4) ## using adam optimization
        
        ## Randomly initialize actor network
        self.actor = Actor(obs_size, act_size, hidden_size, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4) ## using adam optimization
        
	#### The target networks are time-delayed copies of their original networks that slowly track the learned networks. Using these target value networks greatly improve stability in learning. Here’s why: In methods that do not use target networks, the update equations of the network are interdependent on the values calculated by the network itself, which makes it prone to divergence. ####
	
        ## Randomly initialize critic target network
        self.critic_target = Critic(obs_size, act_size, hidden_size).to(device)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        ## Randomly initialize actor target
        self.actor_target = Actor(obs_size, act_size, hidden_size, max_action).to(device)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        
        ## Randomly initialize replay buffer
        self.replay_buffer = deque(maxlen=1000000)

    def act(self, state, train=True):
        action = self.actor.act(state, self.device)
        if train:
            ## Select action with exploration noise.
            action = (
                action + np.random.normal(0, self.exploration_noise, size=self.act_size)
            ).clip(-self.max_action, self.max_action)
        return action

    def update_parameters(self, batch_size=256):
        if len(self.replay_buffer) < batch_size:
            return
	## Select a batch of random sample from replay buffer
        batch = random.sample(self.replay_buffer, k=batch_size)
        state, action, reward, next_state, not_done = [torch.FloatTensor(t).to(self.device) for t in zip(*batch)]

        # Update critic
        with torch.no_grad():
            ## For continuous action spaces, exploration is done via adding noise to the action itself.
            noise = (torch.randn_like(action)*self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            
            ## original q value calculated using value network, not the target value network.
            q_original = self.critic(state, action)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            q_next = self.critic_target(next_state, next_action)
            
            ## updated q value is calculated using target q network.
            ## The value network is updated similarly as is done in Q-learning. The updated Q value is obtained by the Bellman equation
            q_target = reward + not_done * self.gamma * q_next


	## We minimize the mean-squared loss between the updated Q value and the original Q value.
        critic_loss = F.mse_loss(q_original, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

	## For the policy function, our objective is to maximize the expected return
        ## To calculate the policy loss, we take the derivative of the objective function with respect to the policy parameters. Take the mean of the sum of gradients calculated from the mini-batch.
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
	    
	## Update target critic(value) network
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_((1.0-self.tau)*target_param.data + self.tau*param.data)
	    
	## Update target actor(policy) network
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_((1.0-self.tau)*target_param.data + self.tau*param.data)
           
    def save(self, directory, name):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, name))
        torch.save(self.actor_target.state_dict(), '%s/%s_actor_target.pth' % (directory, name))
        
        torch.save(self.critic.state_dict(), '%s/%s_crtic_1.pth' % (directory, name))
        torch.save(self.critic_target.state_dict(), '%s/%s_critic_1_target.pth' % (directory, name))
        
    def load(self, directory, name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        self.critic.load_state_dict(torch.load('%s/%s_crtic_1.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.critic_target.load_state_dict(torch.load('%s/%s_critic_1_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
               
    def load_actor(self, directory, name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
