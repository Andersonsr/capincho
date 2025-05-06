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
    parser.add_argument('--random_seed', type=int, default=777, help='random seed for qualitative evaluation')
    parser.add_argument('--num_images', '-n', type=int, default=10, help='number of images to evaluate')

    # generation arguments
    parser.add_argument('--top_p', type=float, default=None, help='sampling top-p')
    parser.add_argument('--top_k', type=int, default=None, help='sampling top-k')
    parser.add_argument('--do_sample', action='store_true', default=False, help='')
    parser.add_argument('--num_beams', type=int, default=1, help='number of beams')
    parser.add_argument('--max_tokens', type=int, default=200, help='maximum number of generated tokens')
    parser.add_argument('--temperature', type=float, default=1, help='logit temperature')
    parser.add_argument('--penalty', type=float, default=None, help='penalty for contrastive search')
    args = parser.parse_args()

    logger = logging.getLogger('captioning')
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    # which dataset to load
    if args.dataset == 'petro':
        embeddings_path_1 = 'embeddings/petrofeatures_openclip.pkl'
        embeddings_path_2 = 'embeddings/petrofeatures_resize_openclip.pkl'

        data1 = PetroDataset(embeddings_path_1, split=args.split)
        data2 = PetroDataset(embeddings_path_2, split=args.split)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path1 = '/nethome/recpinfo/users/fibz/data/results/petro-vis-opt2-cassiana/experiment.json'
    model_path2 = '/nethome/recpinfo/users/fibz/data/results/petro-vis-resized-opt2-cassiana/experiment.json'

    model1 = model_from_json(model_path1, device)
    model1.eval()

    model2 = model_from_json(model_path2, device)
    model2.eval()

    random.seed(args.random_seed)

    generated1 = []
    generated2 = []
    gt = []
    ids = []

    for i in tqdm([random.randint(0, len(data1)) for i in range(args.num_images)]):
        # print(data[i]['image_embeddings'].shape)
        embedding1 = data1[i]['image_embeddings']
        embedding2 = data2[i]['image_embeddings']

        logging.debug(f'loaded embedding 1 shape: {embedding1.shape}')
        logging.debug(f'loaded embedding 2 shape: {embedding2.shape}')

        output = model1.caption(embedding1,
                               max_tokens=args.max_tokens,
                               top_k=args.top_k,
                               top_p=args.top_p,
                               num_beams=args.num_beams,
                               temperature=args.temperature,
                               sample=args.do_sample,
                               penalty=args.penalty)
        generated1.append(output[0])

        output = model2.caption(embedding2,
                                max_tokens=args.max_tokens,
                                top_k=args.top_k,
                                top_p=args.top_p,
                                num_beams=args.num_beams,
                                temperature=args.temperature,
                                sample=args.do_sample,
                                penalty=args.penalty)
        generated2.append(output[0])

        if 'image_id' in data1[i].keys():
            ids.append(data1[i]['image_id'])

        gt.append(data1[i]['captions'])

    for i in range(len(gt)):
        if len(ids) > 0:
            print('ID: ', ids[i])

        if type(gt[i]) is list:
            print('ORIGINAL: ', gt[i][0])
        else:
            print('ORIGINAL: ' + gt[i])

        print('GENERATED: ', generated1[i])
        print('GENERATED (preprocessed): ', generated2[i])
        print()


