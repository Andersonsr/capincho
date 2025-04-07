import argparse
import random
import torch
from embeddingsDataset import PetroDataset, COCODataset
from decoder import model_from_json
from tqdm import tqdm
from textLoader import TextLoader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-e', '--embeddings', type=str, required=True, help='path to embeddings file')
    parser.add_argument('-m', '--model', type=str, default='experiments/checkpoint.json', required=True,
                        help='path to experiment json file')
    parser.add_argument('-s', '--split', type=str, default='val', choices=['train', 'val'],
                        help='split to load for evaluation')
    parser.add_argument('--random_seed', type=int, default=777, help='random seed for qualitative evaluation')
    parser.add_argument('--num_images', '-n', type=int, default=10, help='number of images to evaluate')
    parser.add_argument('--dataset', type=str, required=True, choices=['petro', 'petro-txt', 'coco'])

    args = parser.parse_args()
    if args.dataset == 'petro':
        data = PetroDataset(args.embeddings, split=args.split)
    elif args.dataset == 'petro-txt':
        data = TextLoader(args.embeddings, split=args.split, has_embeddings=True)
    elif args.dataset == 'coco':
        data = COCODataset(args.embeddings, n_captions=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_from_json(args.model, device)
    model.eval()
    random.seed(args.random_seed)

    generated = []
    gt = []
    ids = []
    for i in tqdm([random.randint(0, len(data)) for i in range(args.num_images)]):
        # print(data[i]['image_embeddings'].shape)
        if args.dataset == 'petro' or args.dataset == 'coco':
            embedding = data[i]['image_embeddings']
        elif args.dataset == 'petro-txt':
            embedding = data[i]['text_embeddings']
        else:
            raise ValueError(f'{args.dataset} is not a valid dataset')

        generated.append(model.caption(embedding, max_tokens=200, )[0])
        ids.append(data[i]['image_id'])
        gt.append(data[i]['captions'])

    for i in range(len(ids)):
        print('id: ', data[i]['image_id'])
        if type(data[i]['captions']) is list:
            print('ORIGINAL: ', data[i]['captions'][0])
        else:
            print('ORIGINAL: ' + data[i]['captions'])

        print('GENERATED: ', generated[0])


