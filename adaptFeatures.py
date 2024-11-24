import argparse
import os
import pickle
import json
import torch
from adapters import ContrastiveResidualAdapter, SigAdapter
from embeddingsDataset import COCODataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "")


def adapt_features(model,
                   embeddings_path='datasets_torchvision/embeddings/coco_ViTL_val.pkl',
                   save_path='datasets_torchvision/embeddings/coco_MPT.pkl',):

    dataset = COCODataset(embeddings_path)
    # single batch
    loader, indices = dataset.get_loader(batch_size=len(dataset), shuffle=False)
    for batch in loader:
        images = model.image_projection(batch['image_embeddings']).detach().cpu()
        texts = model.text_projection(batch['texts_embeddings']).detach().cpu()

        data = {'image_embeddings': images,
                'texts_embeddings': texts,
                'image_id': dataset[:]['image_id'],
                'image_name': dataset[:]['image_name'],}


        # print(data['image_id'])
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', '-n', type=str, required=True, help='experiment name to load')
    parser.add_argument('--output', '-o', type=str, required=True, help='output file path')
    parser.add_argument('--embeddings', '-e', type=str, required=True, help='embeddings file path')
    args = parser.parse_args()

    with open(args.experiment, 'r') as f:
        config = json.load(f)

    logit_scale = config['logit_scale'] * torch.ones([])

    if config['adapter'] == 'contrastive':
        model = ContrastiveResidualAdapter(config['embedding_dim'], config['alpha'], logit_scale,
                                           config['learnable_alpha'])

    else:
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

