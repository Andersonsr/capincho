import pickle
import torch
from torch.utils.data import Dataset
import argparse
import numpy as np


class TextLoader(Dataset):
    def __init__(self, data_path, split='train'):

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        lim = int(0.9 * len(data['text_embeddings']))
        # print(f'LEN: {lim}')
        if split == 'train':
            self.embeddings = data['text_embeddings'][:lim]
            self.texts = data['captions'][:lim]

        if split == 'val':
            self.embeddings = data['text_embeddings'][lim:]
            self.texts = data['captions'][lim:]

    def __len__(self):
        return len(self.texts)

    def collate_fn(self, batch):
        text_embeddings = []
        captions = []
        for e in batch:
            text_embeddings.append(e['text_embeddings'])
            captions.append(e['captions'])

        return {'text_embeddings': torch.stack(text_embeddings),
                'captions': captions}

    def __getitem__(self, index):
        embedding = self.embeddings[index]
        return {'captions': self.texts[index], 'text_embeddings': embedding}

    def get_loader(self, batch_size=32):
        indices = np.arange(len(self.texts))
        sampler = torch.utils.data.SequentialSampler(indices)
        loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=sampler, shuffle=False,
                                             collate_fn=self.collate_fn)
        return loader, indices


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='path to embeddings pkl')
    args = parser.parse_args()
    data = TextLoader(args.path)
    loader, indices = data.get_loader(batch_size=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from trainDecoder import prepare_batch

    for batch in loader:
        print(batch['text_embeddings'].shape)
        print(len(batch['captions']))
        batch = prepare_batch(batch, True, device, num_descriptions=1)
        print(batch['embeddings'].shape)
        break
