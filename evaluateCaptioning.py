import argparse
import logging
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
    parser.add_argument('-s', '--split', type=str, default='all', choices=['train', 'val'],
                        help='split to load for evaluation, if all embeddings are in the same file')
    parser.add_argument('--random_seed', type=int, default=777, help='random seed for qualitative evaluation')
    parser.add_argument('--num_images', '-n', type=int, default=10, help='number of images to evaluate')
    parser.add_argument('--dataset', type=str, required=True, choices=['petro', 'petro-txt', 'coco', 'cego'])
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
    parser.add_argument('--patched', action='store_true', default=False, help='use patched embeddings')
    # generation arguments
    parser.add_argument('--temperature', type=float, default=1, help='logit temperature')
    parser.add_argument('--penalty_alpha', type=float, default=None, help='penalty for contrastive search')
    parser.add_argument('--diversity_penalty', type=float, default=None,
                        help='diversity penalty for diverse beam search')
    parser.add_argument('--top_k', type=int, default=None, help='sampling top-k')
    parser.add_argument('--do_sample', action='store_true', default=False, help='')
    parser.add_argument('--top_p', type=float, default=None, help='sampling top-p')
    parser.add_argument('--num_beams', type=int, default=1, help='number of beams')
    parser.add_argument('--max_tokens', type=int, default=200, help='maximum number of generated tokens')
    args = parser.parse_args()

    logger = logging.getLogger('captioning')
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    # which dataset to load
    if args.dataset == 'petro':
        data = PetroDataset(args.embeddings, split=args.split)
    elif args.dataset == 'petro-txt':
        data = TextLoader(args.embeddings, split=args.split,)
    elif args.dataset == 'coco':
        data = COCODataset(args.embeddings, n_captions=1)
    elif args.dataset == 'cego':
        data = PetroDataset(args.embeddings, split=None)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DATA: ", data[0])
    model = model_from_json(args.model, device)
    model.eval()
    random.seed(args.random_seed)

    generated = []
    gt = []
    ids = []
    for i in tqdm([random.randint(0, len(data)) for i in range(args.num_images)]):
        # print(data[i]['image_embeddings'].shape)
        if args.dataset == 'petro' or args.dataset == 'coco' or args.dataset == 'cego':
            if args.patched:
                embedding = data[i]['patch_embeddings']
            else:
                embedding = data[i]['image_embeddings']

        elif args.dataset == 'petro-txt':
            embedding = data[i]['text_embeddings']
        else:
            raise ValueError(f'{args.dataset} is not a valid dataset')

        logging.debug(f'loaded embedding shape: {embedding.shape}')
        output = model.caption(embedding,
                               max_tokens=args.max_tokens,
                               top_k=args.top_k,
                               top_p=args.top_p,
                               num_beams=args.num_beams,
                               temperature=args.temperature,
                               do_sample=args.do_sample,
                               penalty_alpha=args.penalty_alpha,
                               diversity_penalty=args.diversity_penalty)
        generated.append(output[0])

        if 'image_id' in data[i].keys():
            ids.append(data[i]['image_id'])

        gt.append(data[i]['captions'])
        print(data[i]['captions'])

    for i in range(len(gt)):
        if 'image_id' in data[i].keys():
            print('id: ', ids[i])
        if type(gt[i]) is list:
            print('ORIGINAL: ', gt[i][0])
        else:
            print('ORIGINAL: ' + gt[i])

        print('GENERATED: ', generated[i])
        print()


