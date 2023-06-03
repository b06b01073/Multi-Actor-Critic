from argparse import ArgumentParser
import matplotlib.pyplot as plt 

from component_agent import ComponentAgent
import env
from memory import EpisodicMemory

def train(args):
    agent = ComponentAgent(args.asset)
    market = env.make(csv_path=args.data_path, start=args.start, end=args.end)

    memory = EpisodicMemory(capacity=args.rmsize, max_train_traj_len=args.exp_traj_len,window_length=args.window_length)
   
    earnings = []


    for i in range(args.epoch):
        obs = market.reset()
        agent.reset_noise()
        total_reward = 0

        for _ in range(args.max_iter):
            action, invested_asset = agent.take_action(obs)
            action_bc, next_obs, earning, terminated, _ = market.step(action, invested_asset)

            # memory.append(action_bc, state0, action, reward, done)
            memory.append(action_bc, next_obs, action, earning,terminated)
            obs = next_obs
            experiences = memory.sample(args.batch_size)

            earnings.append(earning - args.asset)
            agent.update_asset(earning)

            agent.learn(experiences)
            agent.soft_update()

            total_reward += earning

        if terminated:
            break
    
    plt.clf()
    plt.plot(earnings)
    plt.savefig('img/result.jpg')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--data_path', '-d', type=str, default='dataset/^GSPC_2000-01-01_2022-12-31.csv')
    parser.add_argument('--start', '-s', type=str, default='2022-12-15')
    parser.add_argument('--end', '-e', type=str, default='2022-12-30')
    parser.add_argument('--asset', '-a', type=float, default=30000)

    ##### Model Setting #####
    # parser.add_argument('--rnn_mode', default='lstm', type=str, help='RNN mode: LSTM/GRU')
    parser.add_argument('--rnn_mode', default='gru', type=str, help='RNN mode: LSTM/GRU')
    parser.add_argument('--input_size', default=14, type=int, help='num of features for input state')
    parser.add_argument('--seq_len', default=15, type=int, help='sequence length of input state')
    parser.add_argument('--num_rnn_layer', default=2, type=int, help='num of rnn layer')
    parser.add_argument('--hidden_rnn', default=128, type=int, help='hidden num of lstm layer')
    parser.add_argument('--hidden_fc1', default=256, type=int, help='hidden num of 1st-fc layer')
    parser.add_argument('--hidden_fc2', default=64, type=int, help='hidden num of 2nd-fc layer')
    parser.add_argument('--hidden_fc3', default=32, type=int, help='hidden num of 3rd-fc layer')
    parser.add_argument('--init_w', default=0.005, type=float, help='initialize model weights') 
    
    ##### Learning Setting #####
    parser.add_argument('--r_rate', default=0.0001, type=float, help='gru layer learning rate')  
    parser.add_argument('--c_rate', default=0.0001, type=float, help='critic net learning rate') 
    parser.add_argument('--a_rate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--beta1', default=0.3, type=float, help='mometum beta1 for Adam optimizer')
    parser.add_argument('--beta2', default=0.9, type=float, help='mometum beta2 for Adam optimizer')
    parser.add_argument('--sch_step_size', default=16*150, type=float, help='LR_scheduler: step_size')
    parser.add_argument('--sch_gamma', default=0.5, type=float, help='LR_scheduler: gamma')
    parser.add_argument('--bsize', default=100, type=int, help='minibatch size')
    
    ##### RL Setting #####
    parser.add_argument('--warmup', default=100, type=int, help='only filling the replay memory without training')
    parser.add_argument('--discount', default=0.95, type=float, help='future rewards discount rate')
    parser.add_argument('--a_update_freq', default=3, type=int, help='actor update frequecy (per N steps)')
    parser.add_argument('--Reward_max_clip', default=15., type=float, help='max DSR reward for clipping')
    parser.add_argument('--tau', default=0.002, type=float, help='moving average for target network')
    ##### original Replay Buffer Setting #####
    parser.add_argument('--rmsize', default=12000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')  
    ##### Exploration Setting #####
    parser.add_argument('--ou_theta', default=0.18, type=float, help='noise theta of Ornstein Uhlenbeck Process')
    parser.add_argument('--ou_sigma', default=0.3, type=float, help='noise sigma of Ornstein Uhlenbeck Process') 
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu of Ornstein Uhlenbeck Process') 
    parser.add_argument('--epsilon_decay', default=100000, type=int, help='linear decay of exploration policy')
    
    ##### Training Trajectory Setting #####
    parser.add_argument('--exp_traj_len', default=16, type=int, help='segmented experiece trajectory length')  
    parser.add_argument('--train_num_episodes', default=2000, type=int, help='train iters each episode')  
    ### Also use in Test (Evaluator) Setting ###
    parser.add_argument('--max_episode_length', default=240, type=int, help='the max episode length is 240 minites in one day')  
    parser.add_argument('--test_episodes', default=243, type=int, help='how many episode to perform during testing periods')
    
    ##### Other Setting #####
    parser.add_argument('--seed', default=627, type=int, help='seed number')
    # parser.add_argument('--date', default=629, type=int, help='date for output file name')
    parser.add_argument('--save_threshold', default=20, type=int, help='lack margin stop ratio')
    parser.add_argument('--lackM_ratio', default=0.7, type=int, help='lack margin stop ratio')
    parser.add_argument('--debug', default=True, dest='debug', action='store_true')
    parser.add_argument('--checkpoint', default="checkpoints", type=str, help='Checkpoint path')
    parser.add_argument('--logdir', default='log')
    # parser.add_argument('--mode', default='test', type=str, help='support option: train/test')
    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')

    parser.add_argument('--data_path', '-d', type=str, default='TX_data/TX_TI.csv')
    parser.add_argument('--start', '-s', type=str, default='2010-01-04')
    parser.add_argument('--end', '-e', type=str, default='2023-12-30')
    parser.add_argument('--asset', '-a', type=float, default=1000000)


    ##### Behavior Cloning #####
    parser.add_argument('--is_BClone', default=True, action='store_true', help='conduct behavior cloning or not')
    parser.add_argument('--is_Qfilt', default=True, action='store_true', help='conduct Q-filter or not')
    parser.add_argument('--use_Qfilt', default=100, type=int, help='set the episode after warmup to use Q-filter')
    parser.add_argument('--lambda_Policy', default=0.7, type=int, help='The weight for actor loss')
    # parser.add_argument('--lambda_BC', default=0.5, type=int, help='The weight for BC loss after Q-filter, default is equal to (1-lambda_Policy)')
 

    
    args = parser.parse_args()

    train(args)