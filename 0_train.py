import sys
sys.path.append('utilities')
import os
import torch
import json
import transformers
from transformers import AutoTokenizer, AutoModel
import numpy as np
from model_creator import model_creator
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers import LlamaTokenizer
from trl import SFTTrainer
import argparse

def read_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sourcefiles', type=str, default='xxxx.json,xxxx1.json', help='training file list')
    parser.add_argument('--outputdir', type=str, default='yyyy.json', help='output dir')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf', help='model name')
    parser.add_argument('--max_steps', type=int, default=5000, help='Training max steps')
    parser.add_argument('--save_steps', type=int, default=1000, help='Save model for each steps')
    parser.add_argument('--batch_size', type=int, default=32, help='Training/Eval batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='gradient_accumulation_steps')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning_rate')
    # parser.add_argument('--instruction', type=str, default='instruction.txt', help='instruction file location')
    # parser.add_argument('--triever', type=str, default='facebook/contriever', help='retriver name')
    # parser.add_argument('--trainyes', action='store_true') #with --train true, without false
    args = parser.parse_args()

    # output_dir = os.path.dirname(args.outputdir)
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir, exist_ok=True)
    return args


def formatting_func(example):
  if example.get("inputsentence", "") != "":
      input_prompt = (f"Below is an instruction that describes a task, paired with an input that provides further context. "
      "Write a response that appropriately completes the request.\n\n"
      "### Instruction:\n"
      f"{example['instruction']}\n\n"
      f"### Input: \n"
      f"{example['inputsentence']}\n\n"
      f"### Response: \n"
      f"{example['response']}")

  else:
    input_prompt = (f"Below is an instruction that describes a task. "
      "Write a response that appropriately completes the request.\n\n"
      "### Instruction:\n"
      f"{example['instruction']}\n\n"
      f"### Response:\n"
      f"{example['response']}")

  return {"text" : input_prompt}


def prepare_data(args):
    files = args.sourcefiles.split(',')
    for file in files:
        print(file)

    datasets = [load_dataset('json', data_files='datasets/'+file+'.json')['train'] for file in files]
    # trainingset = datasets[0].concatenate(*datasets[1:])
    trainingset = concatenate_datasets(datasets)


    # for file in files:
    #     data = load_dataset("json", data_files=file)
    #     formatted_data = data.map(formatting_func)

    #     print( formatted_data["train"])
    #     trainingset += formatted_data
    return trainingset.shuffle(seed=42).map(formatting_func)



def train_model(args, trainingset):

    qlora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer, base_model = model_creator(args.model_name, bnb_config)
    print("3"*50)


    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_special_tokens({'end_token': '[END]'})

    print("4"*50)

    supervised_finetuning_trainer = SFTTrainer(
        base_model,
        train_dataset=trainingset,
        # eval_dataset=test,
        args=transformers.TrainingArguments(
            # per_device_train_batch_size=args.batch_size,
            #per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            # max_steps=args.max_steps,
            num_train_epochs = 3,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            output_dir=args.outputdir,
            optim="paged_adamw_8bit",
            fp16=True,
            #evaluation_strategy = "steps",
            #eval_steps = 1000,
            # save_steps = args.save_steps,
            #load_best_model_at_end=True,
            save_strategy='epoch',
        ),
        tokenizer=tokenizer,
        peft_config=qlora_config,
        dataset_text_field="text",
        max_seq_length=4098
    )
    # supervised_finetuning_trainer = SFTTrainer(
    #     base_model,
    #     train_dataset=trainingset,
    #     # eval_dataset=test,
    #     args=transformers.TrainingArguments(
    #         per_device_train_batch_size=args.batch_size,
    #         #per_device_eval_batch_size=args.batch_size,
    #         gradient_accumulation_steps=args.gradient_accumulation_steps,
    #         learning_rate=args.learning_rate,
    #         max_steps=args.max_steps,
    #         max_grad_norm=0.3,
    #         warmup_ratio=0.03,
    #         output_dir=args.outputdir,
    #         optim="paged_adamw_8bit",
    #         fp16=True,
    #         #evaluation_strategy = "steps",
    #         #eval_steps = 1000,
    #         save_steps = args.save_steps,
    #         #load_best_model_at_end=True,
    #         save_strategy='steps',
    #     ),
    #     tokenizer=tokenizer,
    #     peft_config=qlora_config,
    #     dataset_text_field="text",
    #     max_seq_length=4098
    # )

    # 在训练过程中添加调试信息
    # for i, batch in enumerate(supervised_finetuning_trainer.get_train_dataloader()):
    #     print(f"Batch {i}: {batch}")
    #     if i == 0:
    #         inputs = {k: v.to('cuda') for k, v in batch.items()}
    #         outputs = base_model(**inputs)
    #         print(f"Output shape: {outputs[0].shape}")


    supervised_finetuning_trainer.train()



def main():
    args = read_argument()
    # print(args)
    trainingset = prepare_data(args)
    print(len(trainingset))
    train_model(args, trainingset)




if __name__== '__main__':
    main()



