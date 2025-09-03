import math
import random
import time
import matplotlib.pyplot as plt
import numpy
from tqdm import tqdm
import glob
import cv2 as cv
import pandas as pd
import pickle
import torch
import json
import os
import numpy as np
from util import VALID_LABELS
from pycocotools.coco import COCO

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


def mimic_chunk_labels(filename):
    '''
    edit chunk file with reorganized labels
    :param filename: chunk pkl file to edit
    :return: None
    '''
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        for i, e in enumerate(data['labels']):
            # old labels: positive: 1, negative: 0, uncertain: -1, ignore: nan
            # new labels: positive: 1, negative: 0, uncertain: 2, ignore: 3
            new_labels = {}
            for label in e.keys():
                if math.isnan(e[label]):
                    new_labels[label] = 3
                else:
                    new_labels[label] = 2 if e[label] < 0 else e[label]
            data['labels'][i] = new_labels

    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def mimic_embeddings_labels(filename, output_dir):
    '''
    reorganize dataset labels for a embedding file removing all images with no findings
    :param filename: input embeddings pkl
    :param output_dir: output directory to save chunks
    :return: None
    '''
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


def labels_distribution(embeddings_dir):
    labels_per_condition = {}
    for label in VALID_LABELS:
        labels_per_condition[label] = []
    for chunk in glob.glob(os.path.join(embeddings_dir, '*.pkl'))[:1]:
        with open(chunk, 'rb') as f:
            data = pickle.load(f)
            for sample in data['labels']:
                for k, v in sample.items():
                    labels_per_condition[k].append(v)

    df = pd.DataFrame.from_dict(labels_per_condition)
    out_dict = {'condition': [], 'positive': [], 'negative': [], 'uncertain': [], 'not present': []}
    for label in VALID_LABELS:
        counts = df[label].value_counts()
        out_dict['condition'].append(label)
        out_dict['positive'].append(counts[1])
        out_dict['negative'].append(counts[0])
        out_dict['uncertain'].append(counts[2])
        out_dict['not present'].append(counts[3])

    df = pd.DataFrame.from_dict(out_dict)
    df.to_excel('D:\\mimic\\labels-distribution.xlsx', index=False)
    print('saved to D:\\mimic\\labels-distribution.xlsx')
    print(df)


def samples(condition, chunk_file, n=3):
    with open(chunk_file, 'rb') as f:
        data = pickle.load(f)
        # print(data.keys())
        positives = []
        negatives = []
        uncertain = []
        not_present = []
        for i, sample in enumerate(data['labels']):
            if sample[condition] == 1:
                positives.append(data['captions'][i])
            elif sample[condition] == 0:
                negatives.append(data['captions'][i])
            elif sample[condition] == 2:
                uncertain.append(data['captions'][i])
            elif sample[condition] == 3:
                not_present.append(data['captions'][i])
        print(condition)
        print('-----------------positives-----------------')
        for sample in random.sample(positives, n):
            print(sample)
            print()

        print('-----------------negatives-----------------')
        for sample in random.sample(negatives, n):
            print(sample)
            print()

        print('-----------------uncertain-----------------')
        for sample in random.sample(uncertain, n):
            print(sample)
            print()

        print('-----------------not present-----------------')
        for sample in random.sample(not_present, n):
            print(sample)
            print()


def count_length(dirname):
    import logging
    import time
    import gc
    total_len = 0
    cache = {}
    assert os.path.exists(dirname), '{} does not exist'.format(dirname)
    chunks = glob.glob(os.path.join(dirname, "chunk*.pkl"))
    assert len(chunks) > 0, "No chunks found at {}".format(dirname)
    for chunk in tqdm(chunks):
        logging.debug('loading chunk {} ...'.format(chunk))
        starting_time = time.time()
        with open(chunk, 'rb') as f:
            data = None
            gc.collect()
            data = pickle.load(f)
            ending_time = time.time()
            logging.debug('load time: {}'.format(ending_time - starting_time))
            length = len(data['image_name'])

        logging.info('{} length is {}'.format(chunk, length))
        cache[os.path.basename(chunk)] = length
        total_len += length

    with open(os.path.join(dirname, 'data_length.json'), 'w') as f:
        json.dump(cache, f)


def chunk_load_time():
    chunks = glob.glob('D:\\mimic\\processado\\teste\\chunk*.pkl')
    result = []
    if os.path.exists('D:\\mimic\\processado\\teste\\data_length.json'):
        with open('D:\\mimic\\processado\\teste\\data_length.json') as f:
            result = json.load(f)

    for chunk in chunks:
        exist = False

        for d in result:
            if d['chunk'] == os.path.basename(chunk):
                exist = True

        if not exist:
            print('loading chunk {} ...'.format(chunk))
            start = time.time()
            data = pickle.load(open(chunk, 'rb'))
            end = time.time()
            print('elapsed time', end - start)
            result.append({'chunk': os.path.basename(chunk),
                           'length': len(data['image_name']),
                           'elapsed_time': end - start})

    with open(os.path.join(os.path.dirname(chunks[0]), 'data_length.json'), 'w') as f:
        json.dump(result, f, indent=2)

    for r in result:
        print('chunk size : {} , taxa: {}'.format(r['length'], r['length'] / r['elapsed_time']))


def coco_labels(dataset_root, split):
    coco_objects = COCO(f'{dataset_root}/annotations/instances_{split}2017.json')
    coco_captions = COCO(f'{dataset_root}/annotations/captions_{split}2017.json')

    ids = coco_objects.getImgIds()
    imgs = coco_objects.loadImgs(ids)
    data = {'image_id': [], 'image_name': [], 'labels': [], 'captions': []}
    for i, image in enumerate(tqdm(imgs)):
        ann = coco_objects.loadAnns(coco_objects.getAnnIds(ids[i]))
        labels = []
        for annotation in ann:
            if annotation['category_id'] not in labels:
                labels.append(annotation['category_id'])

        data['image_name'].append(image['file_name'])
        data['image_id'].append(ids[i])
        data['labels'].append(labels)
        ann = coco_captions.loadAnns(coco_captions.getAnnIds(ids[i]))
        texts = [e['caption'] for e in ann]
        data['captions'].append(texts)

    print(len(data['captions']), len(data['image_id']), len(data['image_name']), len(data['labels']))


def coco_results():
    results = {'model': []}
    for model in ['dinov2-opt350m', 'dinov3-opt350m', 'openclip-opt350m']:
        with open(f'checkpoints/{model}/evaluation.json', 'r') as f:
            model_results = json.load(f)
            results['model'].append(model)
            for metric, value in model_results.items():
                metric = metric.replace('_', '-')
                if metric not in results.keys():
                    results[metric] = [value]
                else:
                    results[metric].append(value)

    results['encoder size(M)'] = [86.58, 85.66, 303.97]

    df = pd.DataFrame.from_dict(results)
    df.set_index('model', inplace=True)
    df.to_latex('evaluation.tex')


if __name__ == '__main__':
    # coco_labels('E:/datasets/coco_2017/', 'train')
    coco_results()