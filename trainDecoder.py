import argparse
import pickle
import random
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from embeddingsDataset import COCODataset, PetroDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os
from decoder import Decoder
from textLoader import TextLoader
from util import model_size, learnable_parameters


def prepare_batch(batch, text_only, device, num_descriptions=5):
    '''
    Prepare the batch to be forwarded to the model
    :param batch: batch to be processed
    :param text_only: to use text only or not (Boolean)
    :param device: device to use for computation
    :param num_descriptions: total number of descriptions for each image, used to randomize the captioning
    :return: object with keys 'caption' and 'embeddings'
    '''

    if text_only:
        embeds = batch['text_embeddings']
        embeds = embeds.to(device)
        c = random.randint(0, 4)
        captions = [caption[c] for caption in batch['captions']]
        embeds = [embed[c].unsqueeze(dim=0) for embed in embeds]
        return {'captions': captions, 'embeddings': torch.stack(embeds)}

    else:
        embeds = batch['image_embeddings']
        embeds = embeds.to(device)
        c = random.randint(0, num_descriptions-1)
        captions = [caption[c] for caption in batch['captions']]
        return {'captions': captions, 'embeddings': embeds}


def train(epochs, batch_size, lr, filename, r, alpha, dropout, model_name, prefix_len, fp, text_only,
          full_finetune, schedule, add_noise, variance, save_history, dataset, root, dimension, collapse, log_step,
          normalize):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data
    num_captions = 1
    if dataset == 'coco':
        train_data = COCODataset(f'{filename}', 5)
        val_name = filename.replace('train', 'val')
        val_data = COCODataset(f'{val_name}', 5)
        num_captions = 5

    elif dataset == 'petro':
        train_data = PetroDataset(f'{filename}', split='train')
        val_data = PetroDataset(f'{filename}', split='val')

    else:
        raise ValueError(f'{dataset} is not a valid dataset')

    train_loader, indices = train_data.get_loader(batch_size=batch_size)
    val_loader, indices = val_data.get_loader(batch_size=batch_size)

    train_means = None
    if collapse:
        train_means = train_data.get_text_means().to(device) if text_only else train_data.get_image_means().to(device)

    # model
    decoder = Decoder(model_name, device,
                      prefix_length=prefix_len,
                      precision=fp,
                      add_noise=add_noise,
                      variance=variance,
                      dimension=dimension,
                      collapse=train_means,
                      normalize=normalize)

    if not full_finetune:
        decoder.lora_model(r, alpha, dropout)
        print("Lora model")

    optim = AdamW(decoder.parameters(), lr=lr)

    model_size(decoder)
    learnable_parameters(decoder)

    scheduler = None
    if schedule:
        scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=10,
                                                    num_training_steps=epochs * len(train_loader))

    save_path = os.path.join(args.save_path, 'checkpoint.pt')
    training_losses = []
    validation_losses = []

    # training loop
    for epoch in range(epochs):
        log_loss = []
        i = 0
        for batch in tqdm(train_loader, total=len(train_loader)):
            batch = prepare_batch(batch, text_only, device, num_descriptions=num_captions)
            optim.zero_grad()
            output = decoder(batch)
            output.loss.backward()
            optim.step()

            i += 1
            if schedule:
                scheduler.step()
            loss = output.loss.detach().cpu().item()
            log_loss.append(loss)

            # logging
            if i % log_step == 0 or i == len(train_loader):

                # validation
                log_val_losses = []
                decoder.eval()
                decoder.add_noise = False
                for val_batch in val_loader:
                    val_batch = prepare_batch(val_batch, False, device, num_descriptions=num_captions)
                    if collapse:
                        val_batch['embeddings'] -= train_means

                    with torch.no_grad():
                        val_output = decoder(val_batch)
                        log_val_losses.append(val_output.loss.detach().cpu().item())

                # save step loss and clean list
                validation_losses.append(sum(log_val_losses) / len(log_val_losses))
                training_losses.append(sum(log_loss) / len(log_loss))
                log_loss = []

                # plot and save loss history
                plt.plot(range(len(training_losses)), training_losses, label='training')
                plt.plot(range(len(validation_losses)), validation_losses, label='validation')
                plt.legend()
                plt.xlabel('step')
                plt.ylabel('loss')
                plt.title(f'training loss')

                plt.savefig(f'{root}/loss_plot.png')

                plt.clf()
                log = {'training_loss': training_losses, 'validation_loss': validation_losses}
                with open(f'{root}/loss_log.pkl', 'wb') as f:
                    pickle.dump(log, f)

                decoder.train(True)
                decoder.add_noise = add_noise

        # epoch model
        model_dict = {'epoch': epoch + 1,
                      'model_state_dict': decoder.state_dict(),
                      'optimizer_state_dict': optim.state_dict(),
                      'loss': training_losses[-1]
                      }

        # save models for each epoch or overwrite existing one
        if save_history:
            path = save_path.split('.')[0]
            path += f'_epoch{epoch}.pt'
            torch.save(model_dict, path)
        else:
            torch.save(model_dict, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--embeddings', type=str, default='coco_openCLIP_train', help='embeddings filename')
    parser.add_argument('--rank', type=int, default=16, help='lora rank')
    parser.add_argument('--alpha', type=int, default=32, help='lora alpha parameter')
    parser.add_argument('--dropout', type=float, default=0.05, help='lora dropout parameter')
    parser.add_argument('--model_name', type=str, default="facebook/opt-350m", help='OPT model name')
    parser.add_argument('--prefix_len', type=int, default=10, help='model prefix length')
    parser.add_argument('--fp', choices=['fp16', 'fp32'], default='fp32', help='float precision')
    parser.add_argument('--text_only', action='store_true',
                        help='train using text embeddings as input instead of image embeddings')
    parser.add_argument('--full_finetune', action='store_true', help='fine tune entire model', default=False)
    parser.add_argument('--schedule', action='store_true', help='use linear scheduler', default=False)
    parser.add_argument('--noise', action='store_true', help='add noise to embeddings', default=False)
    parser.add_argument('--variance', type=float, help='variance for noise injection', default=0.016)
    parser.add_argument('--history', action='store_true', help='save epoch history', default=False)
    parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'petro', 'cxr'], help='dataset name')
    parser.add_argument('--save_path', default='/nethome/recpinfo/users/fibz/data/', help='root dir for saving results')
    parser.add_argument('--dimension', default=768, type=int, help='embedding dimension')
    parser.add_argument('--collapse', action='store_true', help='collapse embeddings', default=False)
    parser.add_argument('--normalize', action='store_true', help='normalize embeddings', default=False)
    parser.add_argument('--log_step', type=int, default=5000, help='log step')
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print(f'folders created: {args.save_path}')

    precision = torch.float16 if args.fp == 'fp16' else torch.float32
    train(args.epochs, args.batch_size, args.lr, args.embeddings, args.rank, args.alpha, args.dropout,
          args.model_name, args.prefix_len, precision, args.text_only, args.full_finetune, args.schedule,
          args.noise, args.variance, args.history, args.dataset, args.save_path, args.dimension, args.collapse,
          args.log_step, args.normalize)

    result_dict = args.__dict__
    result_dict['checkpoint_path'] = os.path.join(args.save_path, 'checkpoint.pt')
    with open(f'{args.save_path}/experiment.json', 'w') as f:
        json.dump(result_dict, f, indent=2)
