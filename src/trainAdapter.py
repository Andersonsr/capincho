import argparse
import json
import os.path
import pickle
import time
import torch
from src.adapters import ContrastiveResidualAdapter, SigAdapter
from tqdm import tqdm
from torch.optim import Adam
from src import foundation_models
from dataLoaders import COCODataset

from util import plot_curves
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_training(save_path, batch_size, dataset, model, epochs, lr, patience, delta, restore_best=False):
    train_dataset = COCODataset(dataset)
    val_dataset = COCODataset(dataset.replace('train', 'val'))
    train_loader, train_indices = train_dataset.get_loader(shuffle=False, batch_size=batch_size)
    val_loader, val_indices = val_dataset.get_loader(shuffle=False, batch_size=batch_size)
    save_option = "best" if restore_best else "last"
    if patience < 0:
        patience = epochs

    training_losses = []
    validation_losses = []

    optim = Adam(model.parameters(), lr=lr)

    print(f'training {os.path.basename(save_path)}')
    time.sleep(1)

    for i in tqdm(range(epochs)):
        training_loss = model.train_epoch(train_loader, optim)
        validation_loss = model.val_epoch(val_loader)

        training_losses.append(training_loss)
        validation_losses.append(validation_loss)

        model_dict = {'epoch': i,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optim.state_dict(),
                      'loss': training_losses[-1]
                      }
        torch.save(model_dict, os.path.join(save_path, 'checkpoint.pt'))

    plot_curves(training_losses, validation_losses, os.path.join(save_path, 'loss_plot.png'))
    log = {'training_loss': training_losses, 'validation_loss': validation_losses}
    with open(os.path.join(save_path, 'loss_log.pkl'), 'wb') as f:
        pickle.dump(log, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='openclip', choices=['openclip', 'clip', 'coca'],
                        help='foundation model')
    parser.add_argument('--adapter', type=str, default='contrastive', choices=['contrastive', 'sig'],
                        help='adapter type')
    parser.add_argument('--alpha', type=float, default=0.3, help='residual learning rate')
    parser.add_argument('--bias', type=float, default=-10., help='logit bias, sig adapter')
    parser.add_argument('--embeddings', type=str, required=True,
                        help='training embeddings path')
    parser.add_argument('--use_bias', action='store_true', help='use logit bias in sig adapter', default=False)
    parser.add_argument('--multiple_positives', action='store_true',
                        help='use multiple positives per batch in sig adapter', default=False)
    parser.add_argument('--batch_size', type=int, default=400, help='batch size')
    parser.add_argument('--input_dim', type=int, default=768, help='embedding input dimension')
    parser.add_argument('--learnable_alpha', action='store_true', help='learnable alpha', default=False)
    parser.add_argument('--save_path', type=str, required=True, help='path to save outputs')
    parser.add_argument('--patience', type=int, default=-1, help='early stopping patience, '
                                                                 'negative value means no early stopping')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--best', action='store_true', help='restore best model if using early stopping', default=False)
    parser.add_argument('--delta', type=float, help='minimal improvement for early stopping', default=0.01,)
    parser.add_argument('--epochs', type=int, default=200, help='number training of epochs')
    parser.add_argument('--frozen_text', action='store_true', help='use frozen text encoder', default=False)
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('created directory', args.save_path)

    model_dict = {'coca': foundation_models.OpenCoCa,
                  'clip': foundation_models.CLIP,
                  'openclip': foundation_models.OpenCLIP}

    foundation = model_dict[args.model](device)
    foundation.load_model()

    logit_scale = foundation.backbone.logit_scale

    if args.adapter == 'sig':
        model = SigAdapter(args.input_dim, args.alpha, args.bias, logit_scale, use_logit_bias=args.use_bias,
                           multi_positive=args.multiple_positives,)

    elif args.adapter == 'contrastive':
        model = ContrastiveResidualAdapter(args.input_dim, args.alpha, logit_scale, args.learnable_alpha,
                                           frozen_text=args.frozen_text)
    else:
        raise ValueError('supported adapter types are sig, contrastive')

    model.to(device)
    run_training(args.save_path, args.batch_size, args.embeddings, model, args.epochs, args.lr, args.patience,
                 args.delta, args.best)

    result_dict = args.__dict__
    result_dict['checkpoint_path'] = os.path.join(args.save_path, 'checkpoint.pt')
    result_dict['logit_scale'] = model.logit_scale.detach().cpu().item()

    with open(os.path.join(args.save_path, 'experiment.json'), 'w') as f:
        json.dump(result_dict, f, indent=2)
