import logging
import random
import torch
import torch.nn as nn
from projectionHeads import ResidualLearnableHead, LinearClassificationHead
import numpy as np
from dataLoaders import COCODataset

class ClassificationAdapter(nn.Module):
    def __init__(self, input_dim, initial_residual_ratio, classifiers_names, classifiers_outputs, logit_scale,
                 device, contrastive=False):
        super(ClassificationAdapter, self).__init__()
        self.device = device
        self.imageAdapter = ResidualLearnableHead(input_dim, initial_residual_ratio, False).to(self.device)
        self.contrastive = contrastive
        self.logit_scale = nn.Parameter(logit_scale).to(self.device)
        self.classifiers = {}

        for classifier in classifiers_names:
            self.classifiers[classifier] = LinearClassificationHead(input_dim, classifiers_outputs).to(self.device)

    def forward(self, batch):
        image_embeddings = batch['image_embeddings'].to(self.device, torch.float32).squeeze()
        labels = batch['labels']
        loss_obj = {}
        # resized features logits
        image_embeddings = self.imageAdapter.forward(image_embeddings)
        accumulated_loss = 0
        CE = nn.CrossEntropyLoss(ignore_index=3)
        for classifier in self.classifiers.keys():
            logits = self.classifiers[classifier](image_embeddings)
            loss = CE(logits, labels[classifier].to(self.device))
            logging.debug(f'{classifier} loss: {loss}')
            if np.isnan(loss.cpu().detach().numpy()):
                # all labels are equal to ignore index
                loss = torch.tensor(0.0).to(self.device)
                logging.debug(f'NAN: {labels[classifier]}')

            loss_obj[classifier] = loss
            accumulated_loss += loss

        if self.contrastive:
            text_embeddings = batch['text_embeddings'].to(self.device, torch.float32).squeeze()

            # some datasets have more than one description per image, pick one random description
            c = random.randint(0, text_embeddings.shape[1] - 1)
            text_embeddings = text_embeddings[:, c, :]
            # normalization
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
            # loss computation
            logits = (image_embeddings @ text_embeddings.T) * (self.logit_scale.exp())
            targets = torch.arange(len(batch['image_embeddings'])).to(self.device)
            i_loss = CE(logits, targets)
            t_loss = CE(logits.T, targets)
            contrastive_loss = i_loss + t_loss
            # compounded loss
            accumulated_loss = accumulated_loss + contrastive_loss/2

        logging.debug('accumulated_loss: {}'.format(accumulated_loss))
        loss_obj['loss'] = accumulated_loss
        return loss_obj

    def image_projection(self, embeddings):
        with torch.no_grad():
            return self.imageAdapter(embeddings.to(self.device, torch.float32))

    def text_projection(self, embeddings):
        # this is here just for compatibility with other scripts
        return embeddings.to(self.device, torch.float32)


class ContrastiveResidualAdapter(nn.Module):
    def __init__(self, in_dim, initial_residual_ratio, initial_logit_scale, device, trainable_residual_ratio=True,
                 frozen_text=False, ):
        super(ContrastiveResidualAdapter, self).__init__()
        self.device = device
        self.imageAdapter = ResidualLearnableHead(in_dim, initial_residual_ratio, trainable_residual_ratio)
        self.imageAdapter.to(self.device)

        if not frozen_text:
            self.textAdapter = ResidualLearnableHead(in_dim, initial_residual_ratio, trainable_residual_ratio)
            self.textAdapter.to(self.device)

        self.logit_scale = nn.Parameter(initial_logit_scale).to(self.device)
        self.frozen_text = frozen_text

    def forward(self, batch):
        image_features = batch['image_embeddings'].to(self.device, torch.float32).squeeze().to(self.device)
        text_features = batch['text_embeddings'].to(self.device, torch.float32).to(self.device)

        # some datasets have more than one description per image, pick one random description
        c = random.randint(0, text_features.shape[1]-1)
        text_features = text_features[:, c, :]

        # resized features logits
        image_features = self.imageAdapter.forward(image_features)
        if not self.frozen_text:
            text_features = self.textAdapter.forward(text_features)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # loss computation
        logits = (image_features @ text_features.T) * self.logit_scale.exp()
        targets = torch.arange(len(batch['image_embeddings'])).to(self.device)
        i_loss = nn.CrossEntropyLoss()(logits, targets)
        t_loss = nn.CrossEntropyLoss()(logits.T, targets)
        return {'loss': i_loss + t_loss}

    def image_projection(self, embeddings):
        self.eval()
        return self.imageAdapter(embeddings.to(self.device, torch.float32))

    def text_projection(self, embeddings):
        self.eval()
        return self.textAdapter(embeddings.to(self.device, torch.float32))




