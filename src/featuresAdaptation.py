import argparse
from tqdm import tqdm
import pickle
import json
import torch
from adapters import ContrastiveResidualAdapter, SigAdapter
from embeddingsDataset import COCODataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def adapt_features(model,
                   embeddings_path='datasets_torchvision/embeddings/coco_ViTL_val.pkl',
                   save_path='datasets_torchvision/embeddings/coco_MPT.pkl',):

    dataset = COCODataset(embeddings_path)
    images = []
    texts = []
    for i in tqdm(range(len(dataset))):
        images.append(model.image_projection(dataset.image_embeddings[i]).detach().cpu())
        if not model.frozen_text:
            texts.append(model.text_projection(dataset.text_embeddings[i]).detach().cpu())
        else:
            texts.append(dataset.text_embeddings[i].detach().cpu())

    data = {'image_embeddings': images,
            'text_embeddings': texts,
            'image_id': dataset[:]['image_id'],
            'image_name': dataset[:]['image_name'],
            'captions': dataset[:]['captions']}

    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
        print('Saved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', '-n', type=str, required=True, help='experiment name to load')
    parser.add_argument('--output', '-o', type=str, required=True, help='output file path')
    parser.add_argument('--embeddings', '-e', type=str, required=True, help='embeddings file path')
    args = parser.parse_args()

    with open(args.experiment, 'r') as f:
        config = json.load(f)

    logit_scale = config['logit_scale'] * torch.ones([])
    if 'frozen_text' in config.keys():
        frozen_text = config['frozen_text']
    else:
        frozen_text = False

    if config['adapter'] == 'contrastive':
        model = ContrastiveResidualAdapter(config['input_dim'], config['alpha'], logit_scale,
                                           config['learnable_alpha'], frozen_text=frozen_text)

    elif config['adapter'] == 'sig':
        logit_bias = config['bias'] * torch.ones([])
        model = SigAdapter(config['embedding_dim'], config['alpha'], logit_bias, logit_scale,
                           config['multiple_positives'], config['use_bias'], )

    checkpoint = torch.load(config['checkpoint_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    adapt_features(model,
                   save_path=args.output,
                   embeddings_path=args.embeddings)

