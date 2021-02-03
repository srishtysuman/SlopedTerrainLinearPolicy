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


class Critic(nn.Module):
    """ Twin Q-networks """
    def __init__(self, obs_size, act_size, hidden_size):
        super().__init__()
        self.net1 = MLP(obs_size+act_size, 1, hidden_size)
        self.net2 = MLP(obs_size+act_size, 1, hidden_size)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        return self.net1(state_action), self.net2(state_action)

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


class TD3:
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
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        ## Randomly initialize critic target
        self.critic_target = Critic(obs_size, act_size, hidden_size).to(device)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        ## Randomly initialize actor network
        self.actor = Actor(obs_size, act_size, hidden_size, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        
        ## Randomly initialize actor target
        self.actor_target = Actor(obs_size, act_size, hidden_size, max_action).to(device)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        
        ## Randomly initialize replay buffer
        self.replay_buffer = deque(maxlen=1000000)

    def act(self, state, train=True):
        action = self.actor.act(state, self.device)
        if train:
            ## Select action with exploration noise. A standard step in the markov decision process of the  environment.
            action = (
                action + np.random.normal(0, self.exploration_noise, size=self.act_size)
            ).clip(-self.max_action, self.max_action)
        return action

    def update_parameters(self, batch_size=256):
        if len(self.replay_buffer) < batch_size:
            return
	
	## Sample a minibatch from replay buffer
        batch = random.sample(self.replay_buffer, k=batch_size)
        state, action, reward, next_state, not_done = [torch.FloatTensor(t).to(self.device) for t in zip(*batch)]

        # Update critic
        with torch.no_grad():
            ## For continuous action spaces, exploration is done via adding noise to the action itself.
            noise = (torch.randn_like(action)*self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            q1_next, q2_next = self.critic_target(next_state, next_action)
            q_next = torch.min(q1_next, q2_next)
            
            ## updated q value is calculated using target q network.
            q_target = reward + not_done * self.gamma * q_next

	## We minimize the mean-squared loss between the updated Q value and the original Q value.
        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor at a delay to that of updation of critic(value) network
        if self._timestep % self.policy_freq == 0:
        ## To calculate the policy loss, we take the derivative of the objective function with respect to the policy parameters. Take the mean of the sum of gradients calculated from the mini-batch
            action_new = self.actor(state)
            q1_new, q2_new = self.critic(state, action_new)
            actor_loss = -q1_new.mean()

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
        
