from argparse import ArgumentParser
import pandas as pd

def main(args):
    normalized_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = pd.read_csv(args.read_path)
    normalized_df = df

    for col in df:
        if col in normalized_cols:
            normalized_df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    normalized_df.to_csv(args.write_path)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--read_path', '-r', type=str, default='./TX_data/TX_TI.csv')
    parser.add_argument('--write_path', '-w', type=str, default='./TX_data/Normalized_TX_TI.csv')

    args = parser.parse_args()

    main(args)