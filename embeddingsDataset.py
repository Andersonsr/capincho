import os.path
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class COCODataset(Dataset):
    def __init__(self, path, n_captions=5):
        assert os.path.exists(path), '{} does not exist'.format(path)
        self.text_embeddings = []
        with open(path, 'rb') as f:
            data = pickle.load(f)
            for i in data['texts_embeddings']:
                self.text_embeddings.append(i[:n_captions])
            self.images = data['image_embeddings']
            self.image_id = data['image_id']
            self.image_name = data['image_name']
        # print(self.image_id)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        payload = {'image_id': self.image_id[index],
                   'image_name': self.image_name[index],
                   'image_embeddings': self.images[index],
                   'texts_embeddings': self.text_embeddings[index],}

        return payload

    def get_loader(self, shuffle=False, batch_size=400):
        indices = np.arange(len(self.images))
        if shuffle:
            np.random.shuffle(indices)
        sampler = torch.utils.data.SequentialSampler(indices)
        loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=sampler, shuffle=False)
        return loader, indices


if __name__ == '__main__':
    dataset = COCODataset('embeddings/foundation/coco_openclip_val.pkl')
    dataset2 = COCODataset('embeddings/adapted/coco_openclip_adapter_val.pkl')

    # print(len(dataset[:]['image_embeddings']), len(dataset[:]['texts_embeddings']), len(dataset[:]['captions']))
    # print(len(dataset2[:]['image_embeddings']), len(dataset2[:]['texts_embeddings']), len(dataset2[:]['captions']))
