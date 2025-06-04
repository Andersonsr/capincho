import argparse
import os
from tqdm import tqdm
import pickle
from models.adapters import adapter_from_json
import sys
import torch

# path trick
path = os.path.normpath(os.path.join(os.path.join(os.path.abspath(__file__)), '..', '..'))
sys.path.append(path)
from util import VALID_LABELS
from models.adapters import ContrastiveResidualAdapter, ClassificationAdapter
from data.dataLoaders import COCODataset, MIMICLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def adapt_features(model, dataloader, save_path):
    image_embeddings = []
    text_embeddings = []
    data = {'image_name': [], 'image_id': [], 'captions': [], 'labels': []}

    for batch in tqdm(dataloader):
        image_embeddings.append(model.image_projection(batch['image_embeddings']))
        text_embeddings.append(model.text_projection(batch['text_embeddings']))
        data['image_name'] += batch['image_name']
        data['image_id'] += batch['image_id']
        data['captions'] += batch['captions']
        if 'labels' in batch.keys():
            data['labels'] += batch['labels']

    data['image_embeddings'] = torch.cat(image_embeddings).detach().cpu()
    data['text_embeddings'] = torch.cat(text_embeddings).detach().cpu()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', '-n', type=str, required=True, help='experiment name to load')
    parser.add_argument('--adapter', choices=['contrastive', 'classification'], help='adapter type')
    parser.add_argument('--dataset', choices=['petro', 'coco', 'mimic',], help='dataset to load')
    parser.add_argument('--output', '-o', type=str, required=True, help='output file path')
    parser.add_argument('--embeddings', '-e', type=str, required=True, help='embeddings file path')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size')
    args = parser.parse_args()

    assert os.path.exists(args.embeddings), 'embeddings file does not exist'
    assert not os.path.isdir(args.embeddings), 'embeddings is directory'

    # data loader
    if args.dataset == 'mimic':
        dataset = MIMICLoader(args.embeddings, unchanged_labels=True)

    elif args.dataset == 'coco':
        dataset = COCODataset(args.embeddings)

    else:
        raise ValueError(f'{args.dataset} dataset not implemented')

    model = adapter_from_json(args.experiment)
    model.eval()
    data_loader = dataset.get_loader(args.batch_size)
    adapt_features(model, data_loader, save_path=args.output)

