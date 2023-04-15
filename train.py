from component_agent import ComponentAgent
import env

def train():
    agent = ComponentAgent()
    market = env.make('dataset/^GSPC_2000-01-01_2022-12-31.csv', start='2022-12-15', end='2022-12-30')

    state, _ = market.reset()

    while True:
        action = agent.take_action(state)
        state, reward, terminated, _ = market.step(action)

        agent.learn()

        if terminated:
            break


if __name__ == '__main__':
    train()