from tqdm import tqdm
from PIL import Image
import glob
import cv2 as cv
import pandas as pd
import pickle
import torch
import json
import os
import numpy as np


def pkl_to_xlsx(pkl_file: str, xlsx_file: str):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    data_dict = {}
    for i in range(len(data[0])):
        data_dict[f'max_len {50 + i * 50}'] = []

    for i in range(len(data)):
        for j in range(len(data[i])):
            data_dict[f'max_len {50 + j * 50}'].append(data[i][j])

    df = pd.DataFrame(data_dict)
    df.to_excel(xlsx_file, index=False)


def concatenate_pkls(pkl_files: list, output_file: str):
    data_dict = {'image_features': [], 'labels': []}
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            data_dict['image_features'] += data['image_features']
            data_dict['labels'] += data['label']

    with open(output_file, 'wb') as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def rename_column(pkl_file: str):
    data_dict = {'image_features': [], 'labels': []}
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
        data_dict['image_features'] = data['image_features']
        data_dict['labels'] = data['label']

    with open(pkl_file, 'wb') as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def resize_image(image, size):
    h, w = image.shape[:2]
    max_d = max(h, w)
    if max_d > size:
        ratio = size / max_d
        resized = cv.resize(image, (int(w * ratio), int(h * ratio)))
        n_h, n_w = resized.shape[:2]
        h_delta = size - n_h
        w_delta = size - n_w
        blank_image = np.zeros((size, size, 3), np.uint8)
        blank_image[:, :, :] = 255
        blank_image[int(h_delta / 2):int(h_delta / 2) + n_h, int(w_delta / 2):int(w_delta / 2) + n_w, :] = resized
        return blank_image

    else:
        w_delta = size - w
        h_delta = size - h
        blank_image = np.zeros((size, size, 3), np.uint8)
        blank_image[:, :, :] = 255
        blank_image[int(h_delta / 2):int(h_delta / 2) + h, int(w_delta / 2):int(w_delta / 2) + w, :] = image
        return blank_image


def concat_images():
    imgs = []
    for d in [1, 2, 4, 8, 16]:
        path = f'plots/t1 training/custom adapter k={d} d=1024 aircraft.png'
        imgs.append(cv.imread(path, cv.IMREAD_COLOR))

    upper = np.concatenate((np.ones_like(imgs[1]) * 255, imgs[0], imgs[1]), axis=1)
    lower = np.concatenate((imgs[2], imgs[3], imgs[4]), axis=1)
    final = np.concatenate((upper, lower), axis=0)
    cv.imwrite('final vit.png', final)


def count_mimic():
    files = glob.glob('/mnt/d/datasets/mimic-cxr-jpg/2.1.0/files/p10/*')

    print(len(files))


def generate_dummy_texts(n=36):
    import string
    import random
    import pandas
    dummy_texts = {'captions': [], 'image_embeddings': [], 'text_embeddings': [], 'image_id': []}
    for i in range(n):
        dummy_texts['captions'].append(''.join(random.choice(string.ascii_uppercase) for _ in range(10)))
        dummy_texts['image_id'].append(''.join(random.choice(string.ascii_uppercase) for _ in range(10)))
        dummy_texts['image_embeddings'].append(torch.rand(1, 768))
        dummy_texts['text_embeddings'].append(torch.rand(1, 768))

    with open('dummy_petro.pkl', 'wb') as f:
        pickle.dump(dummy_texts, f)


def fixDataCego():
    dataset_path = '/mnt/d/PraCegoVer/pracegover_400k.json'
    name_root = 'embeddings/foundation/cego_openclip'

    with open(dataset_path, 'rb') as f:
        json_object = json.load(f)
        for split in json_object.keys():
            captions = []
            for image in tqdm(json_object[split]):
                captions.append(image['caption'])

            with open(f'{name_root}_{split}.pkl', 'rb') as out:
                data = pickle.load(out)
                data['captions'] = captions

                with open(f'{name_root}_{split}(1).pkl', 'wb') as fixed:
                    pickle.dump(data, fixed)


def shuffle_pkl(in_path, out_path):
    with open(in_path, 'rb') as f:
        data = pickle.load(f)
        df = pd.DataFrame.from_dict(data)
        df = df.sample(frac=1).reset_index(drop=True)
        data = df.to_dict('list')
        with open(out_path, 'wb') as f2:
            pickle.dump(data, f2)


def mimic_stats(filename, dataset_root):
    with open(filename, 'r') as f:
        data = json.load(f)
        for annotation in data:
            image_name = '/'.join(annotation['image'].split('/')[1:])
            image_path = os.path.join(dataset_root, image_name)
            im = cv.imread(image_path)
            print(im.shape)
            break


if __name__ == '__main__':
    mimic_stats('/')

