from argparse import ArgumentParser
import pandas as pd

def main(args):
    normalized_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA10','STD']
    df = pd.read_csv(args.read_path)
    normalized_df = df
    normalized_df['norm_std5'] = (df['BBAND5UP'] - df['MA5'])
    normalized_df['norm_std5'] = (normalized_df['norm_std5'] - normalized_df['norm_std5'].min()) / (normalized_df['norm_std5'].max() - normalized_df['norm_std5'].min())
    for col in df:
        if col in normalized_cols:
            normalized_df[f'norm_{col}'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
   
            
    normalized_df.to_csv(args.write_path)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    #parser.add_argument('--read_path', '-r', type=str, default='./TX_data/TX_TI.csv')
    #parser.add_argument('--write_path', '-w', type=str, default='./TX_data/Normalized_TX_TI.csv')
    parser.add_argument('--read_path', '-r', type=str, default='./TX_data/^GSPC_2000-01-01_2022-12-31.csv')
    parser.add_argument('--write_path', '-w', type=str, default='./TX_data/normalized_^GSPC_2000-01-01_2022-12-31.csv')

    args = parser.parse_args()

    main(args)