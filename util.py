
def plot_curves(training, validation, output_name):
    import matplotlib.pyplot as plt
    plt.plot(training, label=f'training loss')
    plt.plot(validation, label=f'validation loss')

    plt.text(len(training), training[-1], f'{training[-1]:.3}')
    plt.text(len(validation), validation[-1], f'{validation[-1]:.3}')

    plt.title(f'training loss')
    plt.legend()
    plt.savefig(output_name)
    plt.clf()


def coco_texts():
    import pandas
    import torchvision.datasets as dset
    data = dset.CocoCaptions(root=f'datasets_torchvision/coco_2017/train2017',
                             annFile=f'datasets_torchvision/coco_2017/annotations/captions_train2017.json', )
    texts = []
    for img, caption in data:
        texts += caption[:5]
    data = {'texts': texts}
    df = pandas.DataFrame(data)
    df.to_csv(f'datasets_torchvision/coco_2017/texts.csv', index=False)


def model_size(model):
    import torch
    size_model = 0
    for param in model.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits
    print(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")
    return size_model / 8e6


def learnable_parameters(model):
    learnable = 0
    total = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            learnable += param.numel()

    print(f'total params: {total / 1e6:.2f}M,  learnable params: {learnable / 1e6:.2f}M')


def split_sentence(sentence, limit):
    assert limit > 0, "split sentence, limit must be greater than 0"
    from math import ceil
    lines = ceil(len(sentence) / limit)
    # print(lines)
    new_text = ''
    for i in range(lines):
        # print(i)
        delim = (i + 1) * limit
        ini = i * limit
        if delim < len(sentence):
            new_text += sentence[ini:delim] + '\n'
            # print(sentence[ini:delim])
        else:
            new_text += sentence[ini:]
            # print(sentence[ini:])

    return new_text


def shuffle_pkl(in_path, out_path):
    import pandas as pd
    import pickle
    with open(in_path, 'rb') as f:
        data = pickle.load(f)
        df = pd.DataFrame.from_dict(data)
        df = df.sample(frac=1).reset_index(drop=True)
        data = df.to_dict('list')
        with open(out_path, 'wb') as f2:
            pickle.dump(data, f2)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    args = parser.parse_args()

    shuffle_pkl(args.in_path, args.out_path)

