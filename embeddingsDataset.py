
import os.path
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class COCODataset(Dataset):
    def __init__(self, path, n_captions=5):
        assert os.path.exists(path), '{} does not exist'.format(path)
        self.text_embeddings = []
        self.captions = []
        with open(path, 'rb') as f:
            data = pickle.load(f)
            # print(data.keys())
            for i in range(len(data['text_embeddings'])):
                self.text_embeddings.append(data['text_embeddings'][i][:n_captions])
                self.captions.append(data['captions'][i][:n_captions])

            self.image_embeddings = data['image_embeddings']
            self.image_id = data['image_id']
            self.image_name = data['image_name']

        # print(len(self.text_embeddings), len(self.captions))
        # print(self.image_id)

    def __len__(self):
        return len(self.image_embeddings)

    def __getitem__(self, index):
        payload = {'image_id': self.image_id[index],
                   'image_name': self.image_name[index],
                   'image_embeddings': self.image_embeddings[index],
                   'text_embeddings': self.text_embeddings[index],
                   'captions': self.captions[index]}
        return payload

    def collate_fn(self, batch):
        ids = []
        names = []
        image_embeddings = []
        text_embeddings = []
        captions = []
        for e in batch:
            ids.append(e['image_id'])
            names.append(e['image_name'])
            image_embeddings.append(e['image_embeddings'])
            text_embeddings.append(e['text_embeddings'])
            captions.append(e['captions'])

        return {'image_id': ids,
                'image_name': names,
                'image_embeddings': torch.stack(image_embeddings),
                'text_embeddings': torch.stack(text_embeddings),
                'captions': captions}

    def get_loader(self, shuffle=False, batch_size=400):
        indices = np.arange(len(self.image_embeddings))
        if shuffle:
            np.random.shuffle(indices)
        sampler = torch.utils.data.SequentialSampler(indices)
        loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=sampler, shuffle=False,
                                             collate_fn=self.collate_fn)
        return loader, indices

    def get_image_means(self):
        embeds = torch.stack(self.image_embeddings)
        means = torch.zeros(1, embeds.shape[-1])
        for e in embeds:
            means += e / e.norm(dim=-1, keepdim=True)
        means = means / embeds.shape[0]
        return means

    def get_text_means(self):
        embeds = []
        for e in self.text_embeddings:
            embeds += e
        embeds = torch.stack(embeds)
        means = torch.zeros(1, embeds.shape[-1])
        for e in embeds:
            means += e / e.norm(dim=-1, keepdim=True)
        means = means / embeds.shape[0]
        return means


if __name__ == '__main__':
    dataset = COCODataset('embeddings/foundation/openclip_coco_val.pkl')
    loader, indices = dataset.get_loader(batch_size=12)
    dataset.get_image_means()
    for batch in loader:
        print(batch['text_embeddings'].shape)
        print(batch['image_embeddings'].shape)
        print(len(batch['image_id']))
        print(len(batch['image_name']))
        print(len(batch['captions']))
        print(batch['text_embeddings'].norm(dim=-1).shape)
        b = batch['image_embeddings'] / batch['image_embeddings'].norm(dim=-1)
        break


