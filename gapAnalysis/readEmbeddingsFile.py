import pickle
import numpy as np


def read_embeddings_file(file_name, all_texts=False, normalize=False):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        # reading embeddings
        text_embeddings = [d[0].cpu().detach().numpy() for d in data['texts_embeddings']]
        text_embeddings = np.array(text_embeddings)

        if all_texts:
            text_embeddings = [d[:5].cpu().detach().numpy() for d in data['texts_embeddings']]
            text_embeddings = np.array(text_embeddings)
            r, c, d = text_embeddings.shape
            text_embeddings = text_embeddings.reshape((r * c, d))

        else:
            image_embeddings = [d.squeeze(0).cpu().detach().numpy() for d in data['image_embeddings']]
            image_embeddings = np.array(image_embeddings)

        # norm
        if normalize:
            image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
            text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)

        return image_embeddings, text_embeddings

