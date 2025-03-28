import argparse
import random
import torch
from textLoader import TextLoader
from decoder import model_from_json
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-e', '--embeddings', type=str, default='embeddings/textEmbeddings.pkl', required=True,
                        help='path to embeddings file')
    parser.add_argument('-m', '--model', type=str, default='experiments/checkpoint.json', required=True,
                        help='path to the saved model')
    parser.add_argument('--random_seed', type=int, default=777, help='random seed for qualitative evaluation')
    parser.add_argument('--num_images', '-n', type=int, default=10, help='number of images to evaluate')
    args = parser.parse_args()
    data = TextLoader(args.embeddings, has_embeddings=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_from_json(args.model, device)
    model.eval()
    random.seed(args.random_seed)
    for i in tqdm([random.randint(0, len(data)) for i in range(args.num_images)]):
        generated = model.caption(data[i]['embeddings'], max_tokens=100, )
        print(data[i]['captions'], generated)


