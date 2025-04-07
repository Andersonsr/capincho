import argparse
import torch
from util import model_size, learnable_parameters
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from mapping import Mapper
import math
import copy
from transformers import T5Model, T5ForConditionalGeneration
from transformers import T5Tokenizer
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    print('lora not available')


class Decoder(nn.Module):
    def __init__(self, model_name, device, precision=torch.float16, prefix_length=10, add_noise=True, variance=0.016,
                 dimension=768, normalize=True):
        super(Decoder, self).__init__()
        self.device = device
        print('decoder device {}'.format(device))
        if 'opt' in model_name:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=precision,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        elif 't5' in model_name:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        self.embeddings_layer = copy.deepcopy(self.model.get_input_embeddings())
        self.add_noise = add_noise
        self.variance = variance
        self.hidden_size = self._get_hidden_size()
        self.prefix_length = prefix_length
        self.fp = precision
        self.mapper = Mapper(dimension, self.hidden_size, self.prefix_length).to(dtype=precision)
        self.normalize = normalize

        if self.device:
            self.model.to(self.device)
            self.mapper.to(self.device)
            self.embeddings_layer.to(self.device)

    def caption(self, embeddings, stochastic=False, max_tokens=50, seed=32):
        if stochastic:
            set_seed(seed)
        if self.normalize:
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        # id do token de inicio de frase
        sos = torch.ones((embeddings.shape[0], 1)).to(dtype=torch.long) * 2

        if self.device:
            sos = sos.to(self.device)
        sos = self.embeddings_layer(sos)

        if self.device:
            sos = sos.to(self.device)
            embeddings = embeddings.to(self.device)

        prefix = self.mapper(embeddings.to(dtype=self.fp)).view(-1, self.prefix_length, self.hidden_size)
        prefix = torch.concat([sos, prefix], dim=1)
        generated_ids = self.model.generate(do_sample=stochastic, max_new_tokens=max_tokens, inputs_embeds=prefix)
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def forward(self, batch):
        embeddings = batch['embeddings'].to(dtype=self.fp)
        captions = batch['captions']

        if self.add_noise:
            embeddings = self.noise_injection(embeddings)
        if self.device:
            embeddings = embeddings.to(self.device)
        if self.normalize:
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        prefix_tokens = self.mapper(embeddings).view(-1, self.prefix_length, self.hidden_size)

        captions_emb = self.get_input_embeds(captions).to(dtype=self.fp)

        if self.device:
            captions_emb = captions_emb.to(self.device)
        if len(captions_emb.shape) == 2:
            captions_emb = captions_emb.unsqueeze(0)

        # final shape [batch, sos + prefix + caption len, d_model]
        input_emb = torch.concat([captions_emb[:, :1, :], prefix_tokens, captions_emb[:, 1:, :]], dim=1).to(self.fp)

        # opt ignores -100 labels during loss computation
        labels = self.tokenizer(captions, return_tensors="pt", padding=True).input_ids.to(self.fp)
        # ignore padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100
        # ignore prefix
        ignore = torch.ones(input_emb.shape[0], self.prefix_length + 1) * -100

        if self.device:
            labels = labels.to(self.device)
            ignore = ignore.to(self.device)
            input_emb = input_emb.to(self.device)

        labels = torch.concat([ignore, labels[:, 1:]], dim=1)
        return self.model(inputs_embeds=input_emb, labels=labels.to(torch.long))

    def get_input_embeds(self, prompt):
        if self.device:
            input_ids = self.tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(self.device).squeeze(0)
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt", padding=True).input_ids.squeeze(0)

        return self.embeddings_layer(input_ids)

    def _get_hidden_size(self):
        ids = self.tokenizer("prompt", return_tensors="pt").input_ids.squeeze(0)
        embeddings = self.model.get_input_embeddings()
        if self.device:
            ids = ids.to(self.device)
            embeddings = embeddings.to(self.device)
        return embeddings(ids).shape[1]

    def noise_injection(self, x, ):
        x = x.to('cuda')
        return x + torch.randn(x.shape, device='cuda', dtype=self.fp) * math.sqrt(self.variance)

    def lora_model(self, r, alpha, dropout):
        for param in self.model.parameters():
            param.requires_grad = False
        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",

        )
        self.model = get_peft_model(self.model, config).to(self.fp)


# utility function
def model_from_json(json_file, device):
    import json
    import os
    with open(json_file, 'r') as f:
        config = json.load(f)
    precision = torch.float16 if config['fp'] == 'fp16' else torch.float32

    decoder = Decoder(config['model_name'], device, prefix_length=config['prefix_len'], precision=precision,
                      add_noise=config['text_only'], dimension=config['dimension'])

    # decoder model is not locally trained
    if not config['full_finetune']:
        decoder.lora_model(config['rank'], config['alpha'], config['dropout'])

    checkpoint = torch.load(config['checkpoint_path'], map_location=device)
    decoder.load_state_dict(checkpoint['model_state_dict'])
    decoder.normalize = config['normalize']

    print('loaded model from {}'.format(json_file))
    learnable_parameters(decoder)

    return decoder


if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True, type=str, help='experiment json to load model')
    parser.add_argument('--input', required=True, type=str, )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_from_json(args.experiment, device)
    input_ids = model.tokenizer(args.input, return_tensors='pt').input_ids
    output = model.model.generate(input_ids=input_ids.to(device))
    text = model.tokenizer.decode(output[0], skip_special_tokens=True)
    print(text)
