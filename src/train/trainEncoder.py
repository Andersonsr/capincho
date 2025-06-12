import argparse
import logging
import os
import sys
import torch
import json
from tqdm import tqdm
from torch.optim import Adam
# path trick
path = os.path.normpath(os.path.join(os.path.join(os.path.abspath(__file__)), '..', '..'))
sys.path.append(path)
from util import learnable_parameters, model_size
from data.dataLoaders import MIMICLoader
from util import plot_curves
from models.adapters import ClassificationAdapter
from util import VALID_LABELS
from models.foundation_models import model_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train encoder')
    parser.add_argument('--lora', action='store_true', default=False, help='apply lora to model')
    parser.add_argument('--dataset', type=str, required=True, choices=['coco', 'petro', 'mimic'], help='training dataset')
    parser.add_argument('--data', type=str, required=True, help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--model', type=str, required=True, choices=model_dict.keys())
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--frozen_text', action='store_true', default=False, help='use frozen text')
    parser.add_argument('--embedding_dim', type=int, default=768, help='embedding dimension')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory to save model and logs')
    parser.add_argument('--num_classes', type=int, choices=[3, 4], required=True, help='output dimension of classifiers')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--logging_interval', type=int, default=None,
                        help='how many batches to wait before logging')
    parser.add_argument('--validation_interval', type=int, default=None,
                        help='how many batches to wait before validating')
    parser.add_argument('--debug', action='store_true', default=False, help='log debug messages')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    logging.info('Loading data...')
    if args.dataset == 'mimic':
        train_data = MIMICLoader(args.data, )
        val_data = MIMICLoader(args.data.replace('train', 'dev'), )
    else:
        raise NotImplementedError

    train_dataloader = train_data.get_loader(args.batch_size)
    val_dataloader = val_data.get_loader(args.batch_size)

    logging.debug('number of training batches: {}'.format(len(train_dataloader)))
    logging.debug('number of training examples: {}'.format(len(train_data)))
    logging.debug('number of validation batches: {}'.format(len(val_dataloader)))

    if args.validation_interval is None:
        args.validation_interval = len(train_dataloader)

    if args.logging_interval is None:
        args.logging_interval = len(train_dataloader)

    os.makedirs(args.output_dir, exist_ok=True)
    classification_adapter = ClassificationAdapter(args.embedding_dim, torch.tensor([1.0]), VALID_LABELS,
                                                   args.num_classes, torch.tensor([100.0]), device, contrastive=False,
                                                   identity=True)

    foundation = model_dict[args.model](device)
    foundation.load_model()

    logging.info('backbone size: {}'.format(model_size(foundation.backbone)))
    logging.info('backbone learnable params: {}'.format(learnable_parameters(foundation.backbone)))
    logging.info('adapter size: {}'.format(model_size(classification_adapter)))
    logging.info('adapter learnable params: {}'.format(learnable_parameters(classification_adapter)))

    optim = Adam(list(foundation.backbone.parameters()) + list(classification_adapter.parameters()), lr=args.lr)

    if args.lora:
        # TODO: apply LoRA here
        raise NotImplementedError

    # for logging purposes
    training_loss = []
    validation_loss = []
    classifier_loss = {}
    for classifier in classification_adapter.classifiers.keys():
        classifier_loss[classifier] = []

    for epoch in range(args.epochs):
        step_training_loss = []
        step_validation_loss = []
        # epoch loss per classifier
        step_classifier_loss = {}
        for classifier in classification_adapter.classifiers.keys():
            step_classifier_loss[classifier] = []

        # training loop
        for i, batch in tqdm(enumerate(train_dataloader), desc="Epoch {}".format(epoch)):
            embeddings = foundation.visual_embedding(batch['image_tensor'])
            if len(embeddings.shape) == 2:
                embeddings = embeddings.unsqueeze(dim=1)

            if epoch == 0 and i == 0:
                logging.debug('image shape: {}'.format(batch.image_tensor.shape))
                logging.debug('embedding shape'.format(embeddings.shape))

            batch['image_embeddings'] = embeddings
            loss_obj = classification_adapter(batch)
            optim.zero_grad()
            loss_obj['loss'].backward()
            optim.step()

            # append loss per classifier for logging
            for classifier in classification_adapter.classifiers.keys():
                step_classifier_loss[classifier].append(loss_obj[classifier].tolist())

            # validation loop
            if (i+1) % args.validation_interval == 0 or i+1 == len(train_dataloader):
                for batch in val_dataloader:
                    with torch.no_grad():
                        embeddings = foundation.visual_embedding(batch['image_tensor'])
                        batch['image_embeddings'] = embeddings
                        loss_obj = classification_adapter(batch)
                        step_validation_loss.append(loss_obj['loss'].tolist())

                validation_loss.append(sum(step_validation_loss) / len(step_validation_loss))

            # logging losses
            if (i+1) % args.logging_interval == 0 or i+1 == len(train_dataloader):
                training_loss.append(sum(step_training_loss) / len(step_training_loss))

                log = {'training_loss': training_loss, 'validation_loss': validation_loss}

                for classifier in classification_adapter.classifiers.keys():
                    classifier_loss[classifier].append(
                        sum(step_classifier_loss[classifier]) / len(step_classifier_loss[classifier]))
                    log[classifier + '_loss'] = classifier_loss[classifier]

                with open(os.path.join(args.output_dir, 'loss_log.json'), 'w') as f:
                    json.dump(log, f, indent=4)

                # plot graph with training and validation loss
                plot_curves(training_loss, validation_loss, os.path.join(args.output_dir, 'loss_plot.png'))

            # model checkpointing
            model_dict = {'epoch': epoch,
                          'model_state_dict': foundation.backbone.state_dict(),
                          'optimizer_state_dict': optim.state_dict(),
                          'loss': training_loss[-1]}

            torch.save(model_dict, os.path.join(args.output_dir, 'backbone_checkpoint.pt'))

            model_dict = {'epoch': epoch,
                          'model_state_dict': classification_adapter.state_dict(),
                          'optimizer_state_dict': optim.state_dict(),
                          'loss': training_loss[-1]}

            torch.save(model_dict, os.path.join(args.output_dir, 'classifier_checkpoint.pt'))

    # finito
    result_dict = args.__dict__
    result_dict['checkpoint_path'] = os.path.join(args.output_dir, 'backbone_checkpoint.pt')
    result_dict['classifiers_checkpoint'] = os.path.join(args.output_dir, 'classifier_checkpoint.pt')

    with open(os.path.join(args.output_dir, 'experiment.json'), 'w') as f:
        json.dump(result_dict, f, indent=2)


