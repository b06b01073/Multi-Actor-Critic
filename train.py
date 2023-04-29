from argparse import ArgumentParser
import matplotlib.pyplot as plt 

from component_agent import ComponentAgent
import env

def train(args):
    agent = ComponentAgent(args.asset)
    market = env.make(csv_path=args.data_path, start=args.start, end=args.end)

    state, _ = market.reset()
    earnings = []

    while True:
        action, invested_asset = agent.take_action(state)
        state, reward, terminated, earning,_ = market.step(action, invested_asset)


        earnings.append(earning - args.asset)
        agent.update_asset(earning)
        agent.learn()

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