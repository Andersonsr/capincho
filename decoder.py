import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from mapping import Mapper
from captioningDataset import CaptioningDataset
import math
import copy
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    print('lora not available')


class OPT(nn.Module):
    def __init__(self, model_name, device, precision=torch.float16, prefix_length=10, add_noise=True, variance=0.016):
        super(OPT, self).__init__()
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=precision,
        )
        # copy embeddings layer
        self.embeddings_layer = copy.deepcopy(self.model.get_input_embeddings())
        self.add_noise = add_noise
        self.variance = variance
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.hidden_size = self._get_hidden_size()
        self.prefix_length = prefix_length
        self.fp = precision
        self.mapper = Mapper(768, self.hidden_size, self.prefix_length).to(dtype=precision)

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
        bos = self.embeddings_layer(torch.tensor([2]).to(dtype=torch.long)).unsqueeze(0)

        if self.device:
            bos = bos.to(self.device)
            embeddings = embeddings.to(self.device)

        prefix = self.mapper(embeddings.to(dtype=self.fp)).view(-1, self.prefix_length, self.hidden_size)
        prefix = torch.concat([bos, prefix], dim=1)
        generated_ids = self.model.generate(do_sample=stochastic, max_new_tokens=max_tokens, inputs_embeds=prefix)
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def forward(self, batch):
        embeddings = batch['embeddings'].to(dtype=self.fp)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        captions = batch['captions']
        if self.add_noise:
            embeddings = self.noise_injection(embeddings)
        if self.device:
            embeddings = embeddings.to(self.device)

        prefix_tokens = self.mapper(embeddings).view(-1, self.prefix_length, self.hidden_size)
        captions_emb = self.get_input_embeds(captions).to(dtype=self.fp)
        if self.device:
            captions_emb = captions_emb.to(self.device)
        # print(prefix_tokens.shape, embeddings.shape, captions_emb.shape)

        # [batch, bos + prefix + caption, d_model]
        input_emb = torch.concat([captions_emb[:, :1, :], prefix_tokens, captions_emb[:, 1:, :]], dim=1).to(self.fp)
        # print(input_emb.shape)

        labels = self.tokenizer(captions, return_tensors="pt", padding=True).input_ids.to(self.fp)
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

    decoder = OPT(config['model_name'], device, prefix_length=config['prefix_len'], precision=precision,
                  add_noise=config['text_only'])

    if not config['full_finetune']:
        decoder.lora_model(config['rank'], config['alpha'], config['dropout'])

    checkpoint = torch.load(config['checkpoint_path'])
    decoder.load_state_dict(checkpoint['model_state_dict'])
    return decoder


if '__main__' == __name__:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "")
    dataset = CaptioningDataset('embeddings/coco_openclip_val.pkl')
    loader = dataset.get_loader()
    model = OPT('facebook/opt-350m', device)
    # model.lora_model(16, 32, 0.05)

    model.embeddings_layer.to('cuda')
    size_model = 0
    for param in model.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits
    print(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")

