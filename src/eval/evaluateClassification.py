import argparse
import torch
import sys
import os
from sklearn.metrics import classification_report
from torch.nn.functional import softmax
# path trick
path = os.path.normpath(os.path.join(os.path.join(os.path.abspath(__file__)), '..', '..'))
sys.path.append(path)
from data.dataLoaders import MIMICLoader
from models.adapters import adapter_from_json
from util import VALID_LABELS
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate image classification')
    parser.add_argument('--experiment', type=str, required=True, help='experiment json file')
    parser.add_argument('--embeddings', type=str, required=True, help='path to embeddings file')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--name', required=True, help='name of output file')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "")
    data = MIMICLoader(args.embeddings)
    loader = data.get_loader(args.batch_size)
    adapter = adapter_from_json(args.experiment)

    predictions = {}
    gt = {}
    for label in VALID_LABELS:
        predictions[label] = []
        gt[label] = []

    # test split len(loader) = 116
    for i, batch in enumerate(loader):
        if i >= 116:
            break
        image_features = adapter.image_projection(batch['image_embeddings'])
        labels = batch['labels']
        for classifier in adapter.classifiers:
            logits = adapter.classifiers[classifier].forward(image_features).squeeze(dim=1)
            pred = torch.argmax(softmax(logits, dim=1), dim=1)
            predictions[classifier] += pred.tolist()
            gt[classifier] += labels[classifier].tolist()

    target_names = ['negative', 'positive', 'uncertain', 'not present']
    result_dict = {'condition': [], 'macro f1-score': [], 'macro precision': [], 'macro recall': []}

    for k in VALID_LABELS:
        report = classification_report(gt[k], predictions[k], labels=range(len(target_names)), target_names=target_names, zero_division=0, output_dict=True)
        result_dict['condition'].append(k)
        result_dict['macro f1-score'].append(round(report['macro avg']['f1-score'], 5))
        result_dict['macro recall'].append(round(report['macro avg']['recall'], 5))
        result_dict['macro precision'].append(round(report['macro avg']['precision'], 5))

    output_file = args.name
    if '.xlsx' not in output_file:
        output_file = output_file + '.xlsx'

    save_filename = os.path.join(os.path.dirname(args.experiment), output_file)
    pd.DataFrame(result_dict).to_excel(save_filename, index=False)
    print('saved results to {}'.format(save_filename))

