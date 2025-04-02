import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,

    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    input_ids = tokenizer(args.input, return_tensors='pt').input_ids
    output = model.generate(input_ids=input_ids.to(device), do_sample=True, max_new_tokens=100)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(text)

