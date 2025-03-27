import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from util import model_size, learnable_parameters
import os
import json
import glob


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='facebook/opt-350m')
    parser.add_argument('--dataset', type=str, default='textDatasets/shuffled-petroles.txt')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--output_dir', type=str, help='folder to save all outputs', required=True)
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
        # check last step
        steps = [int(c.split('-')[-1]) for c in checkpoints]
        steps.sort(reverse=True)
        last_step = steps[0]
        check_path = f'{args.output_dir}/checkpoint-{last_step}'

    tokenizer = AutoTokenizer.from_pretrained(args.model, )

    test_data = load_dataset('text',
                             data_files=args.dataset,
                             encoding='utf8',
                             cache_dir=args.output_dir,
                             split='train[:10%]')
    train_data = load_dataset('text',
                              data_files=args.dataset,
                              encoding='utf8',
                              cache_dir=args.output_dir,
                              split='train[10%:]')

    if args.lora:
        config = LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        trainArgs = SFTConfig(
            fp16=args.fp16,
            logging_steps=5000,
            logging_strategy='steps',
            evaluation_strategy='steps',
            learning_rate=args.lr,
            output_dir=args.output_dir,
            save_strategy='steps',
            save_steps=10000,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.accumulate_grad_steps,
            num_train_epochs=args.epochs,
            overwrite_output_dir=True,
            resume_from_checkpoint=check_path if args.resume else False,
            save_total_limit=10,
            ddp_find_unused_parameters=False,
        )
        trainer = SFTTrainer(
            args.model,
            train_dataset=train_data,
            eval_dataset=test_data,
            dataset_text_field="text",
            peft_config=config,
            args=trainArgs,
        )
        model_size(trainer.model)
        learnable_parameters(trainer.model)

        trainer.train(resume_from_checkpoint=args.resume)

    else:
        raise NotImplemented('Full finetune not implemented yet, please use LoRA.')

    # save model
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    result_dict = args.__dict__
    with open(f'{args.output_dir}/experiment.json', 'w') as f:
        json.dump(result_dict, f, indent=2)

