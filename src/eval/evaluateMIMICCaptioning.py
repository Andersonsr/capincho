import json
import os
import sys
import logging
from tqdm import tqdm
from run import main
import torch
from argparse import ArgumentParser
# path trick
path = os.path.normpath(os.path.join(os.path.join(os.path.abspath(__file__)), '..', '..'))
sys.path.append(path)
from models.decoder import model_from_json
from data.dataLoaders import (MIMICLoader)

if __name__ == '__main__':
    parser = ArgumentParser(description='Evaluate MIMIC captioning, a results.json file will be created at model dir')
    parser.add_argument('--embeddings', type=str, required=True, help='path to embeddings directory')
    parser.add_argument('--model', type=str, required=True, help='path to model experiment file')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='generate captions even if results file already exists')
    parser.add_argument('--debug', action='store_true', default=False, help='logging debug mode')
    # parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--max_tokens', type=int, default=400, help='max number of tokens to be generated')
    args = parser.parse_args()

    assert os.path.exists(args.model), 'experiment file does not exist'
    assert os.path.exists(args.embeddings) and os.path.isdir(args.embeddings), \
        'argument --embeddings must be a existing directory'

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('using device: {}'.format(device))

    model = model_from_json(args.model, device)
    data = MIMICLoader(args.embeddings)
    loader = data.get_loader(1)
    results = []
    result_path = os.path.join(os.path.dirname(args.model), 'results.json')

    # text generation
    if not os.path.exists(result_path) or args.overwrite:
        logging.info('generating captions ...')
        for i, batch in enumerate(tqdm(loader)):
            if i >= 4532:
                break
            with torch.no_grad():
                embeddings = batch['image_embeddings'].squeeze(0)
                if i == 0:
                    logging.debug('embedding shape {}'.format(embeddings.shape))

                output = model.caption(embeddings, max_tokens=args.max_tokens)
                sample = {'id': batch['image_id'][0], 'reference': batch['captions'][0], 'prediction': output[0]}
                results.append(sample)

        # save results to json
        result_dict = args.__dict__
        result_dict['generated'] = results
        with open(result_path, 'w') as f:
            json.dump(result_dict, f, indent=4)

    # load previous generated captions
    if len(results) < 0:
        logging.info('loading captions ...')
        with open(result_path, 'r') as f:
            results = json.load(f)['generated']

    # metrics computation
    main(filepath=result_path, output_dir=os.path.dirname(result_path))

