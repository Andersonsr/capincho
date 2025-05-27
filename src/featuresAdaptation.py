import argparse
import os
from tqdm import tqdm
import pickle
import json
import torch
from adapters import ContrastiveResidualAdapter, ClassificationAdapter
from dataLoaders import COCODataset, MIMICLoader
from util import VALID_LABELS

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

    with open(args.experiment, 'r') as f:
        config = json.load(f)

    logit_scale = config['logit_scale'] * torch.ones([])
    if 'frozen_text' in config.keys():
        frozen_text = config['frozen_text']
    else:
        frozen_text = False

    # create model and load checkpoint
    if config['adapter'] == 'contrastive':
        model = ContrastiveResidualAdapter(config['input_dim'], config['alpha'], logit_scale,
                                           device, config['learnable_alpha'], frozen_text=frozen_text)

    if config['adapter'] == 'classification':
        model = ClassificationAdapter(config['input_dim'], config['alpha'], VALID_LABELS, 3, logit_scale, device, False)

    else:
        raise ValueError('{} not implemented'.format(config['adapter']))

    # data loader
    if args.dataset == 'mimic':
        dataset = MIMICLoader(args.embeddings, unchanged_labels=True)

    elif args.dataset == 'coco':
        dataset = COCODataset(args.embeddings)

    else:
        raise ValueError(f'{args.dataset} dataset not implemented')

    data_loader = dataset.get_loader(args.batch_size)

    checkpoint = torch.load(config['checkpoint_path'])
    del checkpoint['model_state_dict']['logit_scale']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    adapt_features(model, data_loader, save_path=args.output)

