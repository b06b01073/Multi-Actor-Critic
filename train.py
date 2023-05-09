from argparse import ArgumentParser
import matplotlib.pyplot as plt 

from component_agent import ComponentAgent
import env
from memory import EpisodicMemory

def train(args):
    agent = ComponentAgent(args.asset)
    market = env.make(csv_path=args.data_path, start=args.start, end=args.end)

    memory = EpisodicMemory(capacity=args.rmsize, max_train_traj_len=args.exp_traj_len,
                            window_length=args.window_length)
    earnings = []

    for i in range(args.epoch):
        obs = market.reset()
        agent.reset_noise()
        total_reward = 0

        for _ in range(args.max_iter):
            action, invested_asset = agent.take_action(obs)
            next_obs, earning, terminated, _ = market.step(action, invested_asset)

            # memory.append(action_bc, state0, action, reward, done)
            memory.append(obs, action, earning, next_obs, terminated)
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
    
    args = parser.parse_args()

    train(args)