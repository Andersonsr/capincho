import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from mapping import Mapper
from captioningDataset import CaptioningDataset
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
                 dimension=768):
        super(Decoder, self).__init__()
        self.device = device
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

        if self.device:
            self.model.to(self.device)
            self.mapper.to(self.device)
            self.embeddings_layer.to(self.device)

    def generate(self, prompt, stochastic=False, max_tokens=50, seed=32):
        if stochastic:
            set_seed(seed)
        if self.device:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        generated_ids = self.model.generate(input_ids, do_sample=stochastic, max_new_tokens=max_tokens, )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

    def caption(self, embeddings, stochastic=False, max_tokens=50, seed=32):
        if stochastic:
            set_seed(seed)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        sos = torch.tensor([2]).to(dtype=torch.long)
        if self.device:
            sos = sos.to(self.device)
        bos = self.embeddings_layer(sos).unsqueeze(0)

        if self.device:
            bos = bos.to(self.device)
            embeddings = embeddings.to(self.device)

        prefix = self.mapper(embeddings.to(dtype=self.fp)).view(-1, self.prefix_length, self.hidden_size)
        prefix = torch.concat([bos, prefix], dim=1)
        generated_ids = self.model.generate(do_sample=stochastic, max_new_tokens=max_tokens, inputs_embeds=prefix)
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def forward(self, batch):
        model = 'opt'
        if type(self.model) == T5ForConditionalGeneration:
            model = 't5'

        embeddings = batch['embeddings'].to(dtype=self.fp)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        captions = batch['captions']
        if self.add_noise:
            embeddings = self.noise_injection(embeddings)
        if self.device:
            embeddings = embeddings.to(self.device)

        prefix_tokens = self.mapper(embeddings).view(-1, self.prefix_length, self.hidden_size)
        # print(prefix_tokens.shape, embeddings.shape, captions_emb.shape)

        # [batch, sos + prefix + caption, d_model]
        if model == 'opt':
            captions_emb = self.get_input_embeds(captions).to(dtype=self.fp)
            if self.device:
                captions_emb = captions_emb.to(self.device)
            if len(captions_emb.shape) == 2:
                captions_emb = captions_emb.unsqueeze(0)
            # print(captions_emb.shape, prefix_tokens.shape)
            input_emb = torch.concat([captions_emb[:, :1, :], prefix_tokens, captions_emb[:, 1:, :]], dim=1).to(self.fp)

        elif model == 't5':
            eos = torch.ones((prefix_tokens.shape[0], 1), dtype=torch.long)
            if self.device:
                eos = eos.to(self.device)

            eos = self.embeddings_layer(eos)
            # [batch, learned embeds + sos, d_model]
            input_emb = torch.concat([prefix_tokens, eos], dim=1).to(self.fp)

        labels = self.tokenizer(captions, return_tensors="pt", padding=True).input_ids.to(self.fp)
        labels[labels == self.tokenizer.pad_token_id] = -100

        # opt ignores -100 labels during loss computation
        ignore = torch.ones(labels.shape[0], self.prefix_length + 1) * -100
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
    with open(json_file, 'r') as f:
        config = json.load(f)
    precision = torch.float16 if config['fp'] == 'fp16' else torch.float32

    decoder = Decoder(config['model_name'], device, prefix_length=config['prefix_len'], precision=precision,
                      add_noise=config['text_only'], dimension=config['dimension'])

    if not config['full_finetune']:
        decoder.lora_model(config['rank'], config['alpha'], config['dropout'])

    checkpoint = torch.load(config['checkpoint_path'])
    decoder.load_state_dict(checkpoint['model_state_dict'])
    return decoder


if '__main__' == __name__:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = CaptioningDataset('embeddings/foundation/coco_openclip_val.pkl')
    loader = dataset.get_loader()

    model = model_from_json('experiments/t5-base_openclip_ft.json', device)
    for batch in loader:
        model(batch)
        break

    # tokens = model.tokenizer('teste de ids')
    # print(tokens)
    # decoded = model.tokenizer.decode(tokens.input_ids)
    # print(decoded)
