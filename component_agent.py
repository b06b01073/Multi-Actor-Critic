import numpy as np
from model import (RNN, Actor, Critic)
from util import *
from random_process import OrnsteinUhlenbeckProcess
from scipy.special import softmax
import torch
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.optim.lr_scheduler as Scheduler

class ComponentAgent:
    def __init__(self, asset,args):
        self.asset = asset
        # TODO: build the nn model
        # self.net = None

        nb_actions = 1
        self.date = args.date
        ##### Create RNN Layer #####
        self.rnn = RNN(args)
        self.rnn_target = RNN(args)
        ##### Create Actor Network #####
        self.actor = Actor(args)
        self.actor_target = Actor(args)
        ##### Create Critic Network #####
        self.critic = Critic(args)
        self.critic_target = Critic(args)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        # Hyper-parameters
        self.is_training = True
        self.rnn_mode = args.rnn_mode
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon_decay
        self.epsilon = 1.0
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        if torch.cuda.is_available() : 
            self.cuda()
            print('USE CUDA')


        ### Optimizer and LR_scheduler ###
        beta1 = args.beta1
        beta2 = args.beta2
        self.rnn_optim = Adam(self.rnn.parameters(), lr=args.r_rate, betas=(beta1, beta2))
        self.rnn_scheduler = Scheduler.StepLR(self.rnn_optim, step_size=args.sch_step_size, gamma=args.sch_gamma)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.c_rate, betas=(beta1, beta2))
        self.critic_scheduler = Scheduler.StepLR(self.critic_optim, step_size=args.sch_step_size, gamma=args.sch_gamma)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.a_rate, betas=(beta1, beta2))
        self.actor_scheduler = Scheduler.StepLR(self.actor_optim, step_size=args.sch_step_size, gamma=args.sch_gamma)


        ### initialized values 
        self.total_policy_loss = 0
        self.critic_loss = 0
        self.BC_loss = 0
        self.BC_loss_Qf = 0



        ##### Behavior Cloning Setting #####
        self.is_BClone = args.is_BClone
        self.is_Qfilt = args.is_Qfilt
        self.use_Qfilt = args.use_Qfilt
        if self.is_BClone:
            self.lambda_Policy = args.lambda_Policy
            self.lambda_BC = 1-self.lambda_Policy
        else:
            self.lambda_Policy = 1
            self.lambda_BC = 1-self.lambda_Policy
        # self.lambda_BC = args.lambda_BC
        self.BC_loss_func = nn.MSELoss(reduce=False)
        # self.BC_loss_func = nn.BCELoss(reduce=False)

        

    def take_action(self, state, noise_enable=True, decay_epsilon=True):
        # TODO: select action based on the model output
        xh, _ = self.rnn(state)
        action = self.actor(xh)
        
        action = to_numpy(action.cpu()).squeeze(0)
        if noise_enable == True:
            action += self.is_training * max(self.epsilon, 0)*self.random_process.sample()
            # original
            # action = softmax(action)
            # np.clip(action, 1, -1)
            action = np.tanh(action)

        # print(action)
        if decay_epsilon:
            self.epsilon -= self.depsilon
        # return action, self.epsilon

        # action = np.random.uniform(low=-1.0, high=1)
        invested_asset = self.asset * np.abs(action)
        self.asset -= invested_asset

        return action, invested_asset
    
    def learn(self,experiences):
        # TODO: update the model params
        t_len = len(experiences)
        # action_bc, state0, action, reward, done = experiences\
        action_bc, state0, action, reward,state1, done = experiences

        for t in range(t_len):

            a_cx = Variable(torch.zeros(self.num_layer, self.batch_size, self.hidden_rnn)).type(FLOAT).cuda()
            a_hx = Variable(torch.zeros(self.num_layer, self.batch_size, self.hidden_rnn)).type(FLOAT).cuda()
            
            action_bc = np.stack((trajectory.action_bc for trajectory in experiences[t]))
            state0 = np.stack((trajectory.state0 for trajectory in experiences[t]))          
            action = np.stack((trajectory.action for trajectory in experiences[t]))
            action = to_tensor(action)
            reward = np.expand_dims(np.stack((trajectory.reward for trajectory in experiences[t])), axis=1)
            reward = to_tensor(reward)
            state1 = np.stack((trajectory.state0 for trajectory in experiences[t+1]))
            state0_cuda = to_tensor(state0).cuda()
            state1_cuda = to_tensor(state1).cuda()

            self.update_critic(state0_cuda, a_hx,a_cx, action, reward, state1_cuda, done,t_len)
            self.update_actor(action_bc,state0_cuda, a_hx,a_cx, action,t_len)

        ##### Learning rate Scheduling #####
        self.rnn_scheduler.step()
        self.critic_scheduler.step()
        self.actor_scheduler.step()

        ##### Apply Q-filter to BC loss #####
        # if (episode-1) >= (self.warmup+self.use_Qfilt):
        #     self.is_Qfilt=True


        self.total_policy_loss = []
        self.critic_loss = []


    def update_critic(self, state0_cuda, a_hx,a_cx, action, reward, state1_cuda, done,t_len):
        # with torch.no_grad():
        if self.rnn_mode == 'lstm':
            xh0, _ = self.rnn(state0_cuda, (a_hx, a_cx))
            current_q = self.critic([xh0, action.cuda()])
            
            with torch.no_grad():
                xh1, _ = self.rnn_target(state0_cuda, (a_hx, a_cx))
                target_action = self.actor_target(xh1)
                target_action = target_action.detach()
                next_q_value = self.critic_target([xh1, target_action])
            
            
        elif self.rnn_mode == 'gru':
            xh0, _ = self.rnn(state0_cuda, a_hx)
            current_q = self.critic([xh0, action.cuda()])
            
            with torch.no_grad():
                xh1, _ = self.rnn_target(state1_cuda, a_hx)
                target_action = self.actor_target(xh1)
                target_action = target_action.detach()
                next_q_value = self.critic_target([xh1, target_action])
                
        target_q = reward + (1-done) * self.discount * next_q_value.cpu()
        
        value_loss = 0
        value_loss = F.smooth_l1_loss(current_q, target_q.cuda())

        value_loss /= t_len # divide by experience length
        value_loss_total += value_loss 

        self.critic_loss.append(value_loss_total)

        ####### update Critic per step ####### 
        self.rnn.zero_grad()
        self.actor.zero_grad()
        self.critic.zero_grad()
        value_loss.backward()
        self.critic_optim.step()
        self.rnn_optim.step()  



    def update_actor(self, action_bc,state0_cuda, a_hx,a_cx, action,t_len):
        if self.rnn_mode == 'lstm':
            xh_b0, _ = self.rnn(state0_cuda, (a_hx, a_cx))
            behavior_action = self.actor(xh_b0)

            ### Behavior Cloning : Estimate actor action ###
            q_action = self.agent.critic([xh_b0, action.cuda()])

            ### Calculate Actor loss based on Q-value ###
            actor_loss = -self.critic([xh_b0, behavior_action])

            ##### Behavior Cloning Loss #####
            if self.is_BClone:
                ### Estimate prophetic action ###
                q_action_bc = self.agent.critic([xh_b0, action_bc.cuda()])
                
                ### Q_filter & BC_loss ###
                BC_loss = self.BC_loss_func(behavior_action, action_bc.cuda())
                BC_loss = torch.sum(BC_loss,dim=1).unsqueeze(1)
                
                Q_filter = torch.gt(q_action_bc, q_action)
                BC_loss_Qf = BC_loss * (Q_filter.detach())
                if self.is_Qfilt:
                    ### modified Policy loss ###
                    policy_loss = (self.lambda_Policy*actor_loss) + (self.lambda_BC*BC_loss_Qf) 
                else:
                    ### modified Policy loss ###
                    policy_loss = (self.lambda_Policy*actor_loss) + (self.lambda_BC*BC_loss)
                    
            else:  ### Original Policy loss ###
                policy_loss = actor_loss

            
        elif self.rnn_mode == 'gru':
            xh_b0, _ = self.rnn(state0_cuda, a_hx)
            behavior_action = self.actor(xh_b0)    


            ### Behavior Cloning : Estimate actor action ###
            q_action = self.agent.critic([xh_b0, action.cuda()]) 

            ### Calculate Actor loss based on Q-value ###
            behavior_action = self.actor(xh_b0)
            actor_loss = -self.critic([xh_b0, behavior_action])
         
            ##### Behavior Cloning Loss #####
            if self.is_BClone:
                ### Estimate prophetic action ###
                q_action_bc = self.agent.critic([xh_b0, action_bc.cuda()])
                
                ### Q_filter & BC_loss ###
                BC_loss = self.BC_loss_func(behavior_action, action_bc.cuda())
                BC_loss = torch.sum(BC_loss,dim=1).unsqueeze(1)
                
                Q_filter = torch.gt(q_action_bc, q_action)
                BC_loss_Qf = BC_loss * (Q_filter.detach())
                if self.is_Qfilt:
                    ### modified Policy loss ###
                    policy_loss = (self.lambda_Policy*actor_loss) + (self.lambda_BC*BC_loss_Qf)
                else:
                    ### modified Policy loss ###
                    policy_loss = (self.lambda_Policy*actor_loss) + (self.lambda_BC*BC_loss)
                    
            else:  ### Original Policy loss ###
                policy_loss = actor_loss
        
        ################## Actor loss calculation ##################

        if self.is_BClone:
            BC_loss /= t_len
            BC_loss_total +=  BC_loss.mean()  #BC loss
            BC_loss_Qf  /= t_len
            BC_loss_Qf_total += BC_loss_Qf.mean()
            actor_loss /= t_len
            actor_loss_total += actor_loss.mean()   #actor loss
        else:
            BC_loss_total = torch.zeros(1)
            BC_loss_Qf_total = torch.zeros(1)
            actor_loss_total = torch.zeros(1)
        
        policy_loss /= t_len # divide by experience length
        policy_loss_total += policy_loss.mean()

        self.total_policy_loss.append(policy_loss_total)

        ####### Update Actor ###########
        self.rnn.zero_grad()
        self.actor.zero_grad()
        self.critic.zero_grad()
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()
        self.rnn_optim.step()  


        self.actor_loss = actor_loss_total.item()
        self.BC_loss = BC_loss_total.item()
        self.BC_loss_Qf = BC_loss_Qf_total.item()


        
    def soft_update(self):
        ##### Target_Net update #####
        soft_update(self.rnn_target, self.rnn, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)


    def update_asset(self, earning):
        self.asset += earning



    def eval(self):
        self.rnn.eval
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def reset_rnn_hidden(self, done=True):
        self.rnn.reset_hidden_state(done)

    # training : reset_noises
    def reset(self):
        self.random_process.reset_states()

    def cuda(self):
        #device = torch.device('cuda:0')
        self.rnn.cuda()
        self.rnn_target.cuda()
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()
    
    def load_weights(self, checkpoint_path, model_fn):
        if checkpoint_path is None: return False
        
        model_path = checkpoint_path +'/test_case/' +model_fn
        model = torch.load(model_path)
        self.rnn.load_state_dict(model['rnn'])
        self.actor.load_state_dict(model['actor'])
        self.critic.load_state_dict(model['critic'])

        return True

    def save_model(self, checkpoint_path, episode, ewma_reward):
        e_reward = int(np.round(ewma_reward)) #(ewma_reward,2)
        description = '_' +self.rnn_mode +'_' +'ep' +str(episode) +'_' +'rd' +str(e_reward) +'_' +str(self.date) +'.pkl'
        if self.is_BClone:
            description = '_BC' +description
        model_path = checkpoint_path +'/' +description
        torch.save({'rnn': self.rnn.state_dict(),
                    'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict(),
                    # 'actor_target': self.actor_target.state_dict(),
                    # 'critic_target': self.critic_target.state_dict(),
                    # 'rnn_opt': self.rnn_optim.state_dict(),
                    # 'actor_opt': self.actor_optim.state_dict(),
                    # 'critic_opt': self.critic_optim.state_dict(),
                    }, model_path)


