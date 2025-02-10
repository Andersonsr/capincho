import json
import os.path
import random
from util import split_sentence
from embeddingsDataset import COCODataset
from decoder import Decoder, model_from_json
import torch
import argparse
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

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
    args = parser.parse_args()

    decoder = model_from_json(args.experiment, device)
    decoder.eval()

    embeddings = COCODataset(path=args.embeddings, n_captions=1)

    print('\n Evaluating captioning \n')
    coco = COCO('datasets_torchvision/coco_2017/annotations/captions_val2017.json')
    if args.qualitative:
        random.seed(args.random_seed)
        for i in [random.randint(0, len(embeddings)) for i in range(args.num_images)]:
            input_emb = embeddings[i]['image_embeddings'][0].to(device, dtype=decoder.fp)
            generated = decoder.caption(input_emb, max_tokens=100, )
            ann_id = coco.getAnnIds(embeddings[i]['image_id'])
            ann = coco.loadAnns(ann_id)
            text_gt = 'GT: {}\n'.format(ann[0]['caption'])
            text_gen = 'GENERATED: {}\n'.format(generated[0])

            image = Image.open('datasets_torchvision/coco_2017/val2017/{}'.format(embeddings[i]['image_name']))
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
            print(f'generating captions at {name}')
            data = embeddings[:]
            captions = [decoder.caption(torch.tensor(e, dtype=decoder.fp, device=device).squeeze(0))[0] for e in tqdm(data['image_embeddings'], total=len(data['image_embeddings']))]
            results = []
            print(captions)
            for i in range(len(captions)):
                results.append({'image_id': embeddings[i]['image_id'], 'caption': captions[i]})

            with open(name, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            assert os.path.isfile(name), f'{name} does not exist'
            print(f'loading results from {name}')

        res = coco.loadRes(name)
        coco_eval = COCOEvalCap(coco, res)
        coco_eval.evaluate()

        experiment_result = {}
        for metric, score in coco_eval.eval.items():
            experiment_result[metric] = score

        with open(os.path.join(os.path.dirname(args.experiment), 'evaluation.json'), 'w') as f:
            json.dump(experiment_result, f, indent=2)





