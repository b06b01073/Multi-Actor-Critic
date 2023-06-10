from argparse import ArgumentParser
import matplotlib.pyplot as plt 
import numpy as np
from tqdm import tqdm
import torch
from component_agent import ComponentAgent
import env
from memory import EpisodicMemory, ReplayBuffer
from baseline import baseline

def test(testmarket, agent, lastasset):
    agent = ComponentAgent(args)   
    returns = [0]
    assets = [agent.asset]
    obs, _ = testmarket.reset()
    total_reward = 0
    trajectory_steps = 0
    for _ in tqdm(range(len(testmarket))):
        
        filtered_obs = agent.build_state(obs)
        action, invested_asset = agent.take_action(filtered_obs)
        next_obs, reward, terminated, earning, _ = testmarket.step(action, invested_asset,agent.get_freedom())
        total_reward += reward
            
        obs = next_obs



        agent.update_asset(earning)
        assets.append(agent.asset.item())

        trajectory_steps +=1
        if trajectory_steps >= args.exp_traj_len:
        ### 以下設定是為了讓hidden_state繼續往下一個step傳遞 ###
            agent.rnn.reset_hidden_state(done=False)
            trajectory_steps = 0

            #agent.learn(experiences)
            #agent.soft_update()
        if terminated or agent.asset <= 0:
            agent.rnn.reset_hidden_state(done=True)
            break
        
    
    if agent.asset.item()>lastasset:
        lastasset=agent.asset.item()
        agent.save()
    return lastasset, assets
from torch.utils.tensorboard import SummaryWriter
import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def train(args):
    agent = ComponentAgent(args)
    test_asset=0
    simu_asset=0
    Simumarket=env.make(args, train_mode=False)
    testmarket=env.make(args, train_mode=False)
    market = env.make(args, train_mode=True)
    
    #memory = ReplayBuffer(capacity=args.rmsize)
                     # FutureCost=args.FutureCost, FutureFee=args.FutureFee, FutureDFee=args.FutureDfee, FutureTax=args.FutureTax, data_interval=args.#data_interval)

    #memory = ReplayBuffer(capacity=args.rmsize)
    memory = EpisodicMemory(capacity=args.rmsize, max_train_traj_len=args.exp_traj_len,window_length=args.window_length)
   
   #replayMemory = EpisodicMemory(capacity=args.rmsize, max_train_traj_len=args.exp_traj_len,window_length=args.window_length)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
   


    returns = []
    L,S=baseline(args,args.start,args.end)
    lt,st=baseline(args,args.teststart,args.testend)
    slt,sst=baseline(args,args.simustart,args.simuend)
   
    win_this=[]
    win_last=[]
    for i in range(args.epoch):
        obs, _ = market.reset()
        agent.reset_asset()
        settheActionwin = []
        for i in range(0, len(market)):
            settheActionwin.append(np.random.choice([-1, 1], p=[1-args.imitative_Win_Ratio, args.imitative_Win_Ratio]))
        print(settheActionwin)
        #agent.reset_lstm_hidden_state(done=True)
        total_reward = 0
        trajectory_steps = 0

        assets = [agent.asset]

        #actor_hidden_state = np.zeros(args.hidden_dim)
        #critic_hidden_state = np.zeros(args.hidden_dim)

        #total_value_loss = 0
        #total_policy_loss = 0

        count=0
        actionwinnumber=0
        for _ in tqdm(range(len(market)), desc=f'epoch {i}, warm_up {args.warmup}'):

            agent.reset_noise()
            filtered_obs = agent.build_state(obs)
            if i >= args.warmup:
                #action, invested_asset, filtered_obs, new_actor_hidden_state, new_critic_hidden_state = agent.take_action(obs, actor_hidden_state, critic_hidden_state)
                # filtered_obs : state0 : market_observation :
                
                action, invested_asset = agent.take_action(filtered_obs)
            else:
                action = agent.take_example_action(settheActionwin[actionwinnumber],obs)
                
                #if action == 0:
                #    action = np.random.uniform(-0.3, 0.3, (1,)).astype('float32')
                action = np.clip(action, 1, -1)
                #filtered_obs = agent.build_state(obs).squeeze().cpu().numpy()
                invested_asset = args.asset # keep it rolling



            next_obs, reward, terminated, earning, _ = market.step(action, invested_asset, agent.get_freedom())
            actionwinnumber+=1
            if reward > 0: 
                count += 1
                win_this.append(1)
            else: 
                win_this.append(0)
            
            total_reward += reward
            
            memory.append(action_bc,filtered_obs, action.item(), reward, terminated)
            obs = next_obs

            if i >= args.warmup:
                agent.update_asset(earning)
                assets.append(agent.asset.item())
            experiences = memory.sample(args.batch_size)

            trajectory_steps +=1
            if trajectory_steps >= args.exp_traj_len:
                agent.rnn.reset_hidden_state(done=False)
                trajectory_steps = 0
                actor_loss, BC_loss, BC_loss_Qf, tot_policy_loss, critic_loss = agent.learn(experiences,i)
            

            if terminated or agent.asset <= 0:
                agent.rnn.reset_hidden_state(done=True)
                break
            
        if i >= args.warmup-1:
            plt.clf()
            plt.plot(assets)
            plt.plot(L,color='RED')
            plt.plot(S,color='GREEN')
            plt.savefig('img/result.jpg')
            train_asset=agent.asset
            test_asset,test_assets=test(testmarket,agent,test_asset)
            plt.clf()
            plt.plot(test_assets)
            plt.plot(lt,color='RED')
            plt.plot(st,color='GREEN')
            plt.savefig('img/test_result.jpg')
            
            simu_asset,Simu_assets=test(Simumarket,agent,simu_asset)
            plt.clf()
            plt.plot(Simu_assets)
            plt.plot(slt,color='RED')
            plt.plot(sst,color='GREEN')
            plt.savefig('img/Simu_result.jpg')
            agent.increase_action_freedom()
            returns.append(agent.asset / agent.init_asset)
            print(f'epoch: {i}, total_reward: {total_reward}, asset: {train_asset}, return: {train_asset / agent.init_asset}, action_freedom: {agent.action_freedom}, test_return: {test_asset} , Simu_asset: {simu_asset}, Win_rate: {count/len(market)}')
            #print(f'epoch: {i}, total_reward: {total_reward}, asset: {agent.asset}, return: {agent.asset / agent.init_asset}, action_freedom: {agent.action_freedom}')
            logger.add_scalar("total_reward", total_reward, i)
            logger.add_scalar("asset", agent.asset, i)
            logger.add_scalar("return", (agent.asset / agent.init_asset), i)
            returns.append(agent.asset / agent.init_asset)
            logger.add_scalar("actor_loss",actor_loss, i)
            logger.add_scalar("BC_loss", BC_loss, i)
            logger.add_scalar("BC_loss_Qf", BC_loss_Qf, i)
            logger.add_scalar("tot_policy_loss", tot_policy_loss, i)
            logger.add_scalar("critic_loss", critic_loss, i)
            

            # logger.add_scalar("policy_loss", total_policy_loss, i)
            # logger.add_scalar("value_loss", total_value_loss, i)
            win_last=win_this
            plt.clf()
            plt.plot(returns,color='Blue')
            plt.savefig('img/returns.jpg')
        else: print(f'epoch: {i}, total_reward: {total_reward}, count: {count}')
        

        plt.clf()
        plt.plot(returns)
        plt.savefig('img/returns.jpg')

def get_upperbound(market, asset):
    total_reward = 0 
    obs, _ = market.reset()

    max_asset = asset

    while True:
        last_open = obs.iloc[-1]['Open']
        last_close = obs.iloc[-1]['Close']


        if last_open > last_close:
            action = -1
        else:
            action = 1

        action_bc,next_obs, reward, terminated, earning, _ =  market.step(action, asset)

        max_asset += earning

        if terminated:
            break



    obs, _ = market.reset()
    min_asset = asset
    while True:
        last_open = obs.iloc[-1]['Open']
        last_close = obs.iloc[-1]['Close']


        if last_open < last_close:
            action = -1
        else:
            action = 1

        action_bc,next_obs, reward, terminated, earning, _ =  market.step(action, asset)

        min_asset += earning

        if terminated:
            break

    print(f'profit upper_bound is at around {max_asset}, lower_bound is at around {min_asset}')


if __name__ == '__main__':
    parser = ArgumentParser()

    

    ##### Model Setting #####
    #parser.add_argument('--rnn_mode', default='lstm', type=str, help='RNN mode: LSTM/GRU')
    #['norm_Open','norm_Close','norm_High','norm_Low','norm_Volume','norm_MA5', 'norm_MA10']
    parser.add_argument('--input_size', default=7, type=int, help='num of features for input state')
    parser.add_argument('--seq_len', default=15, type=int, help='sequence length of input state')
    parser.add_argument('--num_rnn_layer', default=2, type=int, help='num of rnn layer')
    parser.add_argument('--hidden_rnn', default=128, type=int, help='hidden num of lstm layer')
    parser.add_argument('--hidden_dim', default=512, type=int, help='hidden_dim of gru layer')
    parser.add_argument('--hidden_fc1', default=256, type=int, help='hidden num of 1st-fc layer')
    parser.add_argument('--hidden_fc2', default=64, type=int, help='hidden num of 2nd-fc layer')
    parser.add_argument('--hidden_fc3', default=32, type=int, help='hidden num of 3rd-fc layer')
    parser.add_argument('--init_w', default=0.005, type=float, help='initialize model weights') 
    parser.add_argument('--epoch', default=1000, type=int) 
    parser.add_argument('--agent_type', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_interval', type=int, default=10)
    
    ##### Learning Setting #####
    parser.add_argument('--r_rate', default=0.005, type=float, help='gru layer learning rate')  
    parser.add_argument('--c_rate', default=1e-3, type=float, help='critic net learning rate') 
    parser.add_argument('--a_rate', default=5e-4, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--beta1', default=0.3, type=float, help='mometum beta1 for Adam optimizer')
    parser.add_argument('--beta2', default=0.9, type=float, help='mometum beta2 for Adam optimizer')
    parser.add_argument('--batch_size', default=64, type=int, help='minibatch size')
    parser.add_argument('--warmup', default=10, type=int)
    parser.add_argument('--sch_step_size', default=16*150, type=float, help='LR_scheduler: step_size')
    parser.add_argument('--sch_gamma', default=0.5, type=float, help='LR_scheduler: gamma')
    #parser.add_argument('--bsize', default=100, type=int, help='minibatch size')
    
    ##### RL Setting #####
    parser.add_argument('--discount', default=0.99, type=float, help='future rewards discount rate')
    parser.add_argument('--a_update_freq', default=3, type=int, help='actor update frequecy (per N steps)')
    parser.add_argument('--Reward_max_clip', default=15., type=float, help='max DSR reward for clipping')
    #parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network')
    parser.add_argument('--tau', default=0.002, type=float, help='moving average for target network')
    ##### original Replay Buffer Setting #####
    parser.add_argument('--rmsize', default=100000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')  
    ##### Exploration Setting #####
    parser.add_argument('--ou_theta', default=0.3, type=float, help='noise theta of Ornstein Uhlenbeck Process')
    parser.add_argument('--ou_sigma', default=0.3, type=float, help='noise sigma of Ornstein Uhlenbeck Process') 
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu of Ornstein Uhlenbeck Process') 
    parser.add_argument('--epsilon_decay', default=100000, type=int, help='linear decay of exploration policy')
    
    ##### Training Trajectory Setting #####
    parser.add_argument('--exp_traj_len', default=16, type=int, help='segmented experiece trajectory length')  
    parser.add_argument('--train_num_episodes', default=2000, type=int, help='train iters each episode')  
    ### Also use in Test (Evaluator) Setting ###
    parser.add_argument('--max_episode_length', default=240, type=int, help='the max episode length is 240 minites in one day')  
    parser.add_argument('--test_episodes', default=243, type=int, help='how many episode to perform during testing periods')
    # 
    ##### Other Setting #####
    parser.add_argument('--seed', default=627, type=int, help='seed number')
    parser.add_argument('--date', default=629, type=int, help='date for output file name')
    parser.add_argument('--save_threshold', default=20, type=int, help='lack margin stop ratio')
    parser.add_argument('--lackM_ratio', default=0.7, type=int, help='lack margin stop ratio')
    parser.add_argument('--debug', default=True, dest='debug', action='store_true')
    parser.add_argument('--checkpoint', default="checkpoints", type=str, help='Checkpoint path')
    parser.add_argument('--logdir', default='log')
    # parser.add_argument('--mode', default='test', type=str, help='support option: train/test')
    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    

    parser.add_argument('--data_path', '-d', type=str, default='TX_data/Normalized_TX_TI.csv')
    parser.add_argument('--start', '-s', type=str, default='2022-01-01') # Do not add quote when providing this arguement in command line.
    parser.add_argument('--end', '-e', type=str, default='2022-12-30')
    parser.add_argument('--teststart', '-ts', type=str, default='2010-01-06') # Do not add quote when providing this arguement in command line.
    parser.add_argument('--testend', '-te', type=str, default='2022-12-30')
    parser.add_argument('--simustart', '-ss', type=str, default='2023-01-01') # Do not add quote when providing this arguement in command line.
    parser.add_argument('--simuend', '-se', type=str, default='2023-05-31')
    parser.add_argument('--asset', '-a', type=float, default=1000000)
    
    # future cost
    parser.add_argument('--FutureCost', '-FC', type=float, default=23000)
    parser.add_argument('--FutureTax', '-FT', type=float, default=0.00002)
    parser.add_argument('--FutureFee', '-FF', type=float, default=12)
    parser.add_argument('--FutureDfee', '-FDF', type=float, default=8)
    parser.add_argument('--DotCost','-DC',type=float, default=50)

    ##### Behavior Cloning #####
    parser.add_argument('--is_BClone', default=True, action='store_true', help='conduct behavior cloning or not')
    parser.add_argument('--is_Qfilt', default=True, action='store_true', help='conduct Q-filter or not')
    parser.add_argument('--use_Qfilt', default=100, type=int, help='set the episode after warmup to use Q-filter')
    parser.add_argument('--lambda_Policy', default=0.7, type=int, help='The weight for actor loss')
    # parser.add_argument('--lambda_BC', default=0.5, type=int, help='The weight for BC loss after Q-filter, default is equal to (1-lambda_Policy)')
 
    ##### Imitative Learning #####
    parser.add_argument('--imitative_Win_Ratio', default=0.7, help='Adjust the imitative win ratio')
    #parser.add_argument('--is_Qfilt', default=False, action='store_true', help='conduct Q-filter or not')

    
    args = parser.parse_args()
    args.run_name = "BC_oriReward_n"

    train(args)