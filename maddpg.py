import torch as T
import torch.nn.functional as F
from agent import Agent
from pettingzoo.mpe import simple_adversary_v3
env = simple_adversary_v3.env(N=2, max_cycles=25, continuous_actions=False)
env.reset()

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, 
                 scenario='simple',  alpha=0.01, beta=0.01, fc1=64, 
                 fc2=64, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario
        self.agents = {}
        for agent_idx, agent_name in enumerate(env.possible_agents): # list name of alive agent
            self.agents[agent_name] = Agent(actor_dims[agent_idx],
                                            critic_dims,
                                            n_actions, n_agents,
                                            agent_name = agent_name,
                                            alpha=alpha,
                                            beta=beta,
                                            chkpt_dir=chkpt_dir)


    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent_name, agent in self.agents.items():
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent_name, agent in self.agents.items():
            agent.load_models()

    def choose_action(self, raw_obs): # hàm này BỎ
        actions = []
        for agent_idx, agent in enumerate(self.agents): 
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
        return actions

    def choose_action(self, raw_obs): # self.agents là 1 dict, key = agent_name, value = Agent class
        actions = {agent.agent_name: agent.choose_action(raw_obs[agent.agent_name]) for agent in self.agents.values()} 
        return actions # return dict, keys=agent_name, values=probs distribute

    def learn(self, memory):
        if not memory.ready(): # store > batch
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents['adversary_0'].actor.device

        states = T.tensor(states, dtype=T.float).to(device) # tensor(batch, 28)
        actions = T.tensor(actions, dtype=T.float).to(device) # tensor(3, batch, 5)
        rewards = T.tensor(rewards).to(device)  # tensor(batch, 3)
        states_ = T.tensor(states_, dtype=T.float).to(device) # same states = tensor(batch, 28)
        dones = T.tensor(dones).to(device) # tensor(batch, 3)

        all_agents_new_actions = [] 
        all_agents_new_mu_actions = []
        old_agents_actions = []

        # actor_states - actor_new_states: list with len = n_agents, each elemt with shape = (batch, state_len = 8 or 10)
        for agent_idx,(agent_name, agent) in enumerate(self.agents.items()): # ('adversary_0', Agent())
            new_states = T.tensor(actor_new_states[agent_idx],  # tensor(batch, actor_dim)
                                 dtype=T.float).to(device)
            new_pi = agent.target_actor.forward(new_states) # (batch, 5)

            all_agents_new_actions.append(new_pi)


            mu_states = T.tensor(actor_states[agent_idx],  # tensor(batch, actor_dim)
                                 dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states) # tensor(batch, 5)

            all_agents_new_mu_actions.append(pi)

            old_agents_actions.append(actions[agent_idx]) # (batch, 5)




        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1) # tensor(batch, 15)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1) # tensor(batch, 15)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1) # tensor(batch, 15)




        for agent_idx,(agent_name, agent) in enumerate(self.agents.items()):

            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten() #((b, 28), (b, 15)) -> (b, 1)

            # all column in same row in dones have same value, True or False
            critic_value_[dones[:,0]] = 0.0  # if next state is terminate -> evaluate == 0
            critic_value = agent.critic.forward(states, old_actions).flatten()
            # a từ trải nghiệm

            target = rewards[:,agent_idx].float() + agent.gamma*critic_value_
            # r + gamma * q(next_state, a') với a' được tính từ target actor, q tính từ target critic


            # optimize critic
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True, inputs=list(agent.critic.parameters()))
            agent.critic.optimizer.step()
 
            # optimize actor
            actor_loss = agent.critic.forward(states, mu).flatten() # a tính từ main actor
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True, inputs=list(agent.actor.parameters()))
            agent.actor.optimizer.step()

            agent.update_network_parameters() # target đều đc cập nhật cùng actor, nhưng với 1 step nhỏ