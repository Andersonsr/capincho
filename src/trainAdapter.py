import argparse
import json
import os.path
import logging
import time
from util import VALID_LABELS
import torch
from adapters import ContrastiveResidualAdapter, ClassificationAdapter
from tqdm import tqdm
from torch.optim import Adam
from foundation_models import model_dict
from dataLoaders import COCODataset, MIMICLoader
from util import plot_curves
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_training(save_path, train_loader, val_loader, model, epochs, lr,):
    training_loss = []
    validation_loss = []

    # loss per classifier
    if type(model) is ClassificationAdapter:
        classifier_loss = {}
        for classifier in model.classifiers.keys():
            classifier_loss[classifier] = []

    optim = Adam(model.parameters(), lr=lr)
    print(f'training {os.path.basename(save_path)}')
    time.sleep(1)

    for i in tqdm(range(epochs), desc='training adapter'):
        epoch_training_loss = []
        epoch_validation_loss = []
        # epoch loss per classifier
        if type(model) is ClassificationAdapter:
            epoch_classifier_loss = {}
            for classifier in model.classifiers.keys():
                epoch_classifier_loss[classifier] = []

        for batch in train_loader:
            model.train()
            optim.zero_grad()
            loss_obj = model.forward(batch)
            loss = loss_obj['loss']
            loss.backward()
            optim.step()
            epoch_training_loss.append(loss.tolist())
            logging.debug('training loss: {}'.format(loss))

            # batch loss per classifier
            if type(model) is ClassificationAdapter:
                for classifier in model.classifiers.keys():
                    epoch_classifier_loss[classifier].append(loss_obj[classifier].tolist())

        # validation
        for batch in val_loader:
            model.eval()
            with torch.no_grad():
                loss_obj = model.forward(batch)
                loss = loss_obj['loss']
                epoch_validation_loss.append(loss.tolist())

        # logging
        training_loss.append(sum(epoch_training_loss)/len(epoch_training_loss))
        validation_loss.append(sum(epoch_validation_loss)/len(epoch_validation_loss))
        logging.debug('epoch training loss: {}'.format(training_loss[-1]))
        logging.debug('epoch validation loss: {}'.format(validation_loss[-1]))
        log = {'training_loss': training_loss, 'validation_loss': validation_loss}

        # classification loss logging
        if type(model) is ClassificationAdapter:
            for classifier in model.classifiers.keys():
                classifier_loss[classifier].append(sum(epoch_classifier_loss[classifier])/len(epoch_classifier_loss[classifier]))
                log[classifier+'_loss'] = classifier_loss[classifier]

        with open(os.path.join(save_path, 'loss_log.json'), 'w') as f:
            json.dump(log, f, indent=4)

        # plot graph with training and validation loss
        plot_curves(training_loss, validation_loss, os.path.join(save_path, 'loss_plot.png'))

        # model checkpointing
        model_dict = {'epoch': i,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optim.state_dict(),
                      'loss': training_loss[-1]}
        torch.save(model_dict, os.path.join(save_path, 'checkpoint.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='openclip', choices=model_dict.keys(),
                        help='foundation model')
    parser.add_argument('--adapter', type=str, default='contrastive', help='adapter type',
                        choices=['contrastive', 'classification'],)
    parser.add_argument('--alpha', type=float, default=0.3, help='residual learning rate')
    parser.add_argument('--embeddings', type=str, required=True,
                        help='training embeddings path')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['coco', 'petro', 'petro-txt', 'mimic'])
    parser.add_argument('--batch_size', type=int, default=400, help='batch size')
    parser.add_argument('--input_dim', type=int, default=768, help='embedding input dimension')
    parser.add_argument('--learnable_alpha', action='store_true', help='learnable alpha', default=False)
    parser.add_argument('--save_path', type=str, required=True, help='path to save outputs')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='number training of epochs')
    parser.add_argument('--frozen_text', action='store_true', help='use frozen text encoder', default=False)
    parser.add_argument('--debug', action='store_true', help='debug mode', default=False)
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('created directory', args.save_path)

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    foundation = model_dict[args.model](device)
    foundation.load_model()

    logit_scale = foundation.backbone.logit_scale
    # CREATE ADAPTER
    if args.adapter == 'contrastive':
        model = ContrastiveResidualAdapter(args.input_dim, args.alpha, logit_scale, device, args.learnable_alpha,
                                           frozen_text=args.frozen_text)
    elif args.adapter == 'classification':
        model = ClassificationAdapter(args.input_dim, args.alpha, VALID_LABELS, 3, logit_scale, device)

    else:
        raise ValueError('adapter not supported')

    # LOAD DATASET
    if args.dataset == 'coco':
        train_dataset = COCODataset(args.embeddings)
        val_dataset = COCODataset(args.embeddings.replace('train', 'val'))

    elif args.dataset == 'mimic':
        train_dataset = MIMICLoader(args.embeddings)
        val_dataset = MIMICLoader(args.embeddings.replace('train', 'dev'))

    else:
        raise ValueError('dataset not supported')

    train_loader = train_dataset.get_loader(batch_size=args.batch_size)
    val_loader = val_dataset.get_loader(batch_size=args.batch_size)

    model.to(device)
    # save_path, train_loader, val_loader, model, epochs, lr,
    run_training(args.save_path, train_loader, val_loader, model, args.epochs, args.lr,)

    result_dict = args.__dict__
    result_dict['checkpoint_path'] = os.path.join(args.save_path, 'checkpoint.pt')
    result_dict['logit_scale'] = model.logit_scale.detach().cpu().item()

    with open(os.path.join(args.save_path, 'experiment.json'), 'w') as f:
        json.dump(result_dict, f, indent=2)
