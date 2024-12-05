import argparse
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer
from util import model_size, learnable_parameters
from torch.optim import AdamW
import os
import torch
import glob


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='facebook/opt-350m')
    parser.add_argument('--dataset', type=str, default='textDatasets/publico-COMPLETO.txt')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='logs')
    parser.add_argument('--save_dir', type=str, default='checkpoints/opt-finetune')
    parser.add_argument('--fp16', action='store_true', default=False, help='use 16-bits floating point precision')
    parser.add_argument('--resume', default=False, help='resume from checkpoint', action="store_true")
    parser.add_argument('--lora', action='store_true', default=False, help='Low Rank Adaptation')
    parser.add_argument('--rank', type=int, default=16, help='rank for Low Rank Adaptation')
    parser.add_argument('--alpha', type=float, default=32, help='alpha for Low Rank Adaptation')
    parser.add_argument('--accumulate_grad_steps', type=int, default=1, help='number of steps to accumulate grad')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size per device')
    args = parser.parse_args()

    last_step = 0
    check_path = ''

    if args.resume:
        assert os.path.exists(args.output_dir), 'output directory does not exist'
        checkpoints = glob.glob(f'{args.output_dir}/checkpoint-*')
        assert len(checkpoints) > 0, f'no checkpoints found at {args.output_dir}'
        steps = [int(c.split('-')[-1]) for c in checkpoints]
        steps.sort(reverse=True)
        last_step = steps[0]
        check_path = f'{args.output_dir}/checkpoint-{last_step}'

    tokenizer = AutoTokenizer.from_pretrained(args.model, )

    data = load_dataset('text', data_files=args.dataset, encoding='utf8', cache_dir=args.output_dir)
    data = data.map(lambda sample: tokenizer(sample['text']), batched=True)

    config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        'facebook/opt-350m',
        train_dataset=data['train'],
        dataset_text_field="text",
        peft_config=config,
        args=transformers.TrainingArguments(
            fp16=args.fp16,
            logging_steps=500,
            logging_strategy='steps',
            learning_rate=args.lr,
            output_dir=args.output_dir,
            save_strategy='steps',
            save_steps=500,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.accumulate_grad_steps,
            num_train_epochs=args.epochs,
            overwrite_output_dir=True,
            resume_from_checkpoint=check_path if args.resume else False,
        )
    )

    model_size(trainer.model)
    learnable_parameters(trainer.model)

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.save_dir)

