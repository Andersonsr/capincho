import argparse
import torch
import logging
from util import model_size, learnable_parameters
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from mapping import Mapper
import math
from transformers import T5Model, T5ForConditionalGeneration
from transformers import T5Tokenizer
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    print('lora not available')

logger = logging.getLogger('captioning')


class Decoder(nn.Module):
    def __init__(self, model_name, device, precision=torch.float16, prefix_length=10, add_noise=False, variance=0.016,
                 dimension=768, normalize=False):
        super(Decoder, self).__init__()
        self.device = device
        logging.info('decoder device {}'.format(device))

        if 'opt' in model_name:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=precision,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        elif 't5' in model_name:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)

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

    def caption(self, embeddings, sample=False, max_tokens=200, seed=32, num_beams=1, top_k=None, top_p=None,
                temperature=1.0, penalty=None):
        set_seed(seed)
        if self.normalize:
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        logging.debug(f'input embeddings shape:{embeddings.shape}')
        # id do token de inicio de frase
        sos = torch.ones((embeddings.shape[0], 1)).to(dtype=torch.long) * 2

        if self.device:
            sos = sos.to(self.device)

        embeddings_layer = self.model.get_input_embeddings()
        sos = embeddings_layer(sos)

        if self.device:
            sos = sos.to(self.device)
            embeddings = embeddings.to(self.device)

        prefix = self.mapper(embeddings.to(dtype=self.fp)).view(-1, self.prefix_length, self.hidden_size)
        logging.debug(f'prefix shape: {prefix.shape}')
        prefix = torch.concat([sos, prefix], dim=1)
        logging.debug(f'concatenated shape: {prefix.shape}')

        generated_ids = self.model.generate(do_sample=sample,
                                            max_new_tokens=max_tokens,
                                            inputs_embeds=prefix,
                                            num_beams=num_beams,
                                            top_k=top_k,
                                            top_p=top_p,
                                            temperature=temperature,
                                            penalty_alpha=penalty)

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


        # batch size, patches, model dim
        b, p, d = embeddings.shape
        embeddings = embeddings.view(b*p, 1, d)

        logging.debug(f'forward, input embeddings shape: {embeddings.shape}')
        prefix_tokens = self.mapper(embeddings).view(-1, self.prefix_length, self.hidden_size)
        prefix_tokens = prefix_tokens.view(b, p*self.prefix_length, self.hidden_size)

        logging.debug(f'forward, prefix embeddings shape: {prefix_tokens.shape}')

        captions_emb = self.get_input_embeds(captions).to(dtype=self.fp)
        logging.debug(f'forward, captions embeddings shape: {captions_emb.shape}')

        if self.device:
            captions_emb = captions_emb.to(self.device)

        if len(captions_emb.shape) == 2:
            logging.debug(f'forward, captions embeddings unsqueeze')
            captions_emb = captions_emb.unsqueeze(0)

        # final shape [batch, sos + prefix + caption len, d_model]
        input_emb = torch.concat([captions_emb[:, :1, :], prefix_tokens, captions_emb[:, 1:, :]], dim=1).to(self.fp)
        logging.debug(f'concatenated shape: {input_emb.shape}')

        # opt ignores -100 labels during loss computation
        labels = self.tokenizer(captions, return_tensors="pt", padding=True).input_ids.to(self.fp)
        logging.debug('labels shape: {}'.format(labels.shape))
        # ignore padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100
        # ignore prefix
        ignore = torch.ones(input_emb.shape[0], self.prefix_length*p + 1) * -100
        logging.debug('ignore shape: {}'.format(ignore.shape))
        if self.device:
            labels = labels.to(self.device)
            ignore = ignore.to(self.device)
            input_emb = input_emb.to(self.device)

        labels = torch.concat([ignore, labels[:, 1:]], dim=1)
        logging.debug('final labels shape: {}'.format(labels.shape))
        return self.model(inputs_embeds=input_emb, labels=labels.to(torch.long))

    def get_input_embeds(self, prompt):
        if self.device:
            input_ids = self.tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(self.device).squeeze(0)

        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt", padding=True).input_ids.squeeze(0)

        embeddings_layer = self.model.get_input_embeddings()
        return embeddings_layer(input_ids)

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

    def load_decoder(self, path):
        raise NotImplementedError()


# utility function
def model_from_json(json_file, device):
    import json
    import os
    with open(json_file, 'r') as f:
        config = json.load(f)

    precision = torch.float16 if config['fp'] == 'fp16' else torch.float32
    decoder = Decoder(config['model_name'], device, prefix_length=config['prefix_len'], precision=precision,
                      add_noise=False, dimension=config['dimension'])

    checkpoint = torch.load(config['checkpoint_path'], map_location=device)
    if not os.path.exists(config['model_name']) and not config['full_finetune']:
        # decoder model is on the hub, and is not adapted by default
        decoder.lora_model(config['rank'], config['alpha'], config['dropout'])

    decoder.load_state_dict(checkpoint['model_state_dict'])
    decoder.normalize = config['normalize']

    print('loaded model from {}'.format(json_file))
    learnable_parameters(decoder)

    return decoder


if '__main__' == __name__:
    from embeddingsDataset import COCODataset
    from trainDecoder import prepare_batch
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings',
                        default='embeddings/foundation/openclip_patch_val.pkl',
                        type=str,
                        help='embeddings pkl')
    parser.add_argument('--patch', action='store_true', default=False)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = logging.getLogger('captioning')
    logging.basicConfig(level=logging.DEBUG)
    # model
    decoder = Decoder('facebook/opt-350m', device,
                      prefix_length=2,
                      dimension=768,
                      normalize=True)

    data = COCODataset(args.embeddings, 5)
    print(data[:].keys())
    loader, indices = data.get_loader(batch_size=32)
    for batch in loader:
        batch = prepare_batch(batch, False, args.patch, device)
        print(batch['embeddings'].shape)
        decoder(batch)
        break
