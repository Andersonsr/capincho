import gc
import math

import matplotlib.pyplot as plt
import numpy
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


def mimic_labels(filename, output_dir):
    from util import VALID_LABELS
    to_keep = []
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        data['image_embeddings'] = list(torch.unbind(data['image_embeddings'], dim=0))
        data['text_embeddings'] = list(torch.unbind(data['text_embeddings'], dim=0))

        for i, e in enumerate(tqdm(data['labels'])):
            if 'No Finding' in e and e['No Finding'] != 1:
                # old labels: positive: 1, negative: 0, uncertain: -1, ignore: nan
                # new labels: positive: 1, negative: 0, uncertain: 2, ignore: 3
                new_labels = {}
                for label in VALID_LABELS:
                    if math.isnan(e[label]):
                        new_labels[label] = 3
                    else:
                        new_labels[label] = 2 if e[label] < 0 else e[label]
                data['labels'][i] = new_labels
                to_keep.append(i)

    # initialize dict to store filtered data
    filtered = {}
    for k in data.keys():
        filtered[k] = []

    print('removing unlabeled samples')
    for i in to_keep:
        for k in data.keys():
            filtered[k].append(data[k][i])

    print('len by key', ['{}: {}'.format(k, len(filtered[k])) for k in filtered.keys()])
    print('removed {} samples'.format(len(data['labels']) - len(to_keep)))

    filtered['image_embeddings'] = torch.stack(filtered['image_embeddings'])
    filtered['text_embeddings'] = torch.stack(filtered['text_embeddings'])

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(filename))
    print('saving result to {}'.format(output_path))
    with open(output_path, 'wb') as f2:
        pickle.dump(filtered, f2)


def run_multiple_adaptations():
    import subprocess
    experiment = 'D:\\modelos\\adapters\\mimic-frozentext-openclip-4class\\experiment.json'
    for chunk in glob.glob('D:\\mimic\\processado\\mimic_train_224\\embeddings\\*.pkl'):
        output = chunk.replace('embeddings', 'embeddings_adapter_4class_06')
        print(output)
        command_list = ['python', 'data_processing/featuresAdaptation.py', '--experiment', experiment, '--adapter',
                        'classification',
                        '--dataset', 'mimic', '--output', output, '--embeddings', chunk]
        try:
            result = subprocess.run(command_list)

        except subprocess.CalledProcessError as e:
            print(e.stderr)


def exceed_clip():
    import clip

    with open('D:\\modelos\\decoders\\mimic\\4class-06-1632\\results.json', 'r') as f:
        data = json.load(f)
        lens = []
        exceed = 0
        for sample in data['generated']:
            tokens = clip.tokenize(sample['reference'], context_length=1000, truncate=True)
            lens.append(torch.count_nonzero(tokens, dim=1).tolist()[0])
            if torch.count_nonzero(tokens, dim=1)[0] > 77:
                exceed += 1
    arr = numpy.array(lens)

    print('median: ', np.median(arr))
    print('mean: ', np.mean(arr))
    print('std: ', numpy.std(arr))
    print('exceed: ', exceed / len(lens))


def plot_means():
    data = pickle.load(open('D:\\embeddings\\foundation\\coco\\openclip_val.pkl', 'rb'))
    images = torch.cat(data['image_embeddings'])
    means_f = torch.mean(images, dim=0)

    data = pickle.load(open('D:\\embeddings\\adapted\\coco\\frozentext_openclip_a06_val.pkl', 'rb'))
    images = data['image_embeddings'].squeeze(dim=1)
    means_a = torch.mean(images, dim=0)

    plt.plot(range(len(means_f)), means_f, label='foundation')
    plt.plot(range(len(means_a)), means_a, label='adapted')
    print(torch.mean(means_a), torch.mean(means_f))
    plt.legend()
    plt.show()

