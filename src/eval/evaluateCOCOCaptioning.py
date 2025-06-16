import logging
import json
import os
import sys
import random
import torch
import argparse
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
# path trick
path = os.path.normpath(os.path.join(os.path.join(os.path.abspath(__file__)), '..', '..'))
sys.path.append(path)
from util import split_sentence
from data.dataLoaders import COCODataset
from models.decoder import model_from_json

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f'device used: {device}')
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='experiments/opt-350m-coco.json',
                        help='experiment path')
    parser.add_argument('--embeddings', type=str, default='embeddings/coco_openCLIP_val.pkl')
    parser.add_argument('--qualitative', action='store_true', help='run qualitative evaluation')
    parser.add_argument('--random_seed', type=int, default=777, help='random seed for qualitative evaluation')
    parser.add_argument('--num_images', '-n', type=int, default=10,
                        help='number of images to evaluate in qualitative evaluation')
    parser.add_argument('--load_results', action='store_true', help='load saved results')
    parser.add_argument('--collapse', action='store_true', help='collapse embeddings', default=False)
    parser.add_argument('--debug', action='store_true', help='debug mode', default=False)
    parser.add_argument('--patched', action='store_true', help='use patches embeddings', default=False)
    args = parser.parse_args()

    logger = logging.getLogger('captioning')
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    decoder = model_from_json(args.experiment, device)
    decoder.add_noise = False
    decoder.eval()
    embeddings = COCODataset(path=args.embeddings, n_captions=1)
    means = torch.zeros(len(embeddings))

    if args.collapse:
        train_data = COCODataset(path=args.embeddings.replace('val', 'train'), n_captions=1)
        means = train_data.get_image_means()
        decoder.collapse = means

    logging.info('\n Evaluating captioning \n')
    coco = COCO('D:/datasets_torchvision/coco_2017/annotations/captions_val2017.json')

    if args.qualitative:
        random.seed(args.random_seed)
        for i in [random.randint(0, len(embeddings)) for i in range(args.num_images)]:
            if not args.patched:
                input_emb = embeddings[i]['image_embeddings'][0].to(device, dtype=decoder.fp)
            else:
                input_emb = embeddings[i]['patch_embeddings'][0].to(device, dtype=decoder.fp)

            generated = decoder.caption(input_emb, max_tokens=100, )
            ann_id = coco.getAnnIds(embeddings[i]['image_id'])
            ann = coco.loadAnns(ann_id)
            text_gt = 'GT: {}\n'.format(ann[0]['caption'])
            text_gen = 'GENERATED: {}\n'.format(generated[0])

            image = Image.open('D:/datasets_torchvision/coco_2017/val2017/{}'.format(embeddings[i]['image_name']))
            w, h = image.size[:2]
            font = ImageFont.truetype("fonts/Instruction.ttf", 16)
            lim = int(w / 10)

            new_gt = split_sentence(text_gt, lim)
            new_gen = split_sentence(text_gen, lim)
            new_text = new_gt + new_gen
            lines = new_text.count('\n') + 1

            new_h = h + (lines * 18)
            text_board = Image.new('RGB', (w, new_h - h), (255, 255, 255))
            ImageDraw.Draw(text_board).multiline_text((1, 1), new_text, (0, 0, 0), font=font)

            dst = Image.new('RGB', (w, new_h), (255, 255, 255))
            dst.paste(image, (0, 0))
            dst.paste(text_board, (0, h))
            dst.save('plots/caption/captions_{}'.format(embeddings[i]['image_name']))

    else:
        name = os.path.join(os.path.dirname(args.experiment), 'generated.json')
        if not args.load_results:
            logging.info(f'generating captions at {name}')
            data = embeddings[:]
            if args.patched:
                embeddings = data['patch_embeddings']

            else:
                embeddings = data['image_embeddings']

            logging.debug(f'embedding shape: {embeddings[0].shape}')

            captions = [decoder.caption(e)[0] for e in tqdm(embeddings)]
            results = []
            for i in range(len(captions)):
                results.append({'image_id': data['image_id'][i], 'caption': captions[i]})

            with open(name, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            assert os.path.isfile(name), (f'{name} doe'
                                          f's not exist')
            logging.info(f'loading results from {name}')

        res = coco.loadRes(name)
        coco_eval = COCOEvalCap(coco, res)
        coco_eval.evaluate()

        experiment_result = {}
        for metric, score in coco_eval.eval.items():
            experiment_result[metric] = score

        with open(os.path.join(os.path.dirname(args.experiment), 'evaluation.json'), 'w') as f:
            json.dump(experiment_result, f, indent=2)





