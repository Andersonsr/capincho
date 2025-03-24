import argparse
import pandas as pd
import wget
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Petro')
    parser.add_argument('--url', type=str, required=True, help='url to download images')
    parser.add_argument('--output', type=str, required=True, help='output directory to save the images')
    parser.add_argument('--list', type=str, required=True, help='path to xlsx')

    args = parser.parse_args()
    df = pd.read_excel(args.list)
    for id in tqdm(df['cd_guid'].tolist()):
        wget.download(f'{args.url}/{id}.png', f'{args.output}/{id}.png')
