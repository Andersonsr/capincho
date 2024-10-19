from projectionHeads import TwoLayerHead, ResidualHead
import os.path
import random
import pandas as pd
import pickle
import clip
import torch
from torchinfo import summary
from PIL import Image
import re
from skimage.metrics import structural_similarity as ssim, mean_squared_error, peak_signal_noise_ratio
import lpips
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt


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


def clip_encoders_sizes():
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    models = clip.available_models()
    sizes = []
    for model in models:
        backbone, preprocess = clip.load(model)
        image = preprocess(Image.open("plots/projeto/top_k_acc.png")).unsqueeze(0).to(device, torch.float16)
        summary_data = summary(backbone.visual, input_data=image, verbose=0)
        x = re.search("Trainable params.*", str(summary_data))
        params = x.group(0).split(' ')[-1].replace(',', '')
        print(x.group(0))
        sizes.append(float(params) / 1e6)

    plt.figure()
    plt.bar(models, sizes, align='center')
    plt.title('CLIP Visual encoders learnable parameters')
    plt.xticks(rotation=30)
    plt.ylabel('M learnable parameters')
    plt.tight_layout()
    plt.show()


def resize_image(image, size):
    h, w = image.shape[:2]
    max_d = max(h, w)
    if max_d > size:
        ratio = size / max_d
        resized = cv2.resize(image, (int(w * ratio), int(h * ratio)))
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
        imgs.append(cv2.imread(path, cv2.IMREAD_COLOR))

    upper = np.concatenate((np.ones_like(imgs[1]) * 255, imgs[0], imgs[1]), axis=1)
    lower = np.concatenate((imgs[2], imgs[3], imgs[4]), axis=1)
    final = np.concatenate((upper, lower), axis=0)
    cv2.imwrite('final vit.png', final)


def models_size():
    for d in [256, 512, 768, 1024, 1280, 1536]:
        model = TwoLayerHead(1024, d)
        summary_data = summary(model, (1, 1024), verbose=0)
        x = re.search("Trainable params.*", str(summary_data))
        params = x.group(0).split(' ')[-1].replace(',', '')
        print(d, params)

    model = ResidualHead(1024, 1024, 0.6)
    summary_data = summary(model, (1, 1024), verbose=0)
    x = re.search("Trainable params.*", str(summary_data))
    params = x.group(0).split(' ')[-1].replace(',', '')
    print('residual', params)
