import sys
sys.path.append('utilities')
from model_creator import model_creator

from peft import get_peft_model
import torch
import transformers
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer,LlamaTokenizer
from transformers import AutoTokenizer
from peft import PeftModel
import sentencepiece
import accelerate
import json
import datetime
import ast
import os
import random

import argparse

def get_arguments():
  parser = argparse.ArgumentParser()     
  parser.add_argument('--lora_path', type=str, default='Our_model/', help='LoRA saved path')
  parser.add_argument('--testset', type=str, default='xxx.json', help='Test set file')
  parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-13b-hf', help='model id')
  parser.add_argument('--checkpoints', type=str, default="['1000', '2000', '3000', '4000', '5000']", help='checkpoint list')
  parser.add_argument('--output_dir', type=str, default='Our_model/', help='dir for saving results')
  args = parser.parse_args()
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)
  return args


def load_model_tokenizer_w_saved_lora(args, checkpoint):
  lora_weights= args.lora_path + "/" + checkpoint  #FTOpenLM-just_ourdata   SFTOpenLM-with_ourdata_lolly   SFTOpenLM-Dolly15k
  print(lora_weights)

  # lora_config = LoraConfig.from_pretrained(saved_path)
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16
  )

  base_model = args.model_name
  print(type(lora_weights))

  tokenizer, model = model_creator(base_model, bnb_config)
  # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
  tokenizer.pad_token = tokenizer.eos_token

  # tokenizer=LlamaTokenizer.from_pretrained(base_model)  #, config=config, cache_dir="./llamacache"
  # model = AutoModelForCausalLM.from_pretrained(
  #     base_model,
  #     torch_dtype=torch.float16,
  #     quantization_config=bnb_config,
  #     device_map='auto')

  # model = get_peft_model(model, lora_config)
  model = PeftModel.from_pretrained(
              model,
              lora_weights,
              torch_dtype=torch.float16,
          )

  return model, tokenizer



def make_inference(instruction, context = None):
  if context:
    prompt = f"Below is an instruction that describes a task, paired with an input that provides further context.\n\n### Instruction: \n{instruction}\n\n### Input: \n{context}\n\n### Response: \n"
  else:
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction: \n{instruction}\n\n### Response: \n"
  inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False,max_length=1500).to("cuda:0")
  # outputs = base_model.generate(**inputs, max_new_tokens=100)
  # display(Markdown((tokenizer.decode(outputs[0], skip_special_tokens=True))))
  # model.zero()
  model.eval()
  with torch.no_grad():
      outputs = model.generate(**inputs, max_new_tokens=100,temperature=0.1)
      results=(tokenizer.decode(outputs[0], skip_special_tokens=True))
      return results
      # print(results)
      # print("---- NON-INSTRUCT-TUNED-MODEL ----")

def sample_list(input_list, number_=1000):
    if len(input_list) > number_:
        return random.sample(input_list, number_)
    else:
        return input_list


def gen(input_test_file_name, save_file_name, model, tokenizer):
  fw=open(save_file_name,"w")
  i=0
  with open('datasets/'+input_test_file_name+'.json',"r",encoding="utf-8") as fr:  #path+"test_chuck_final_ICL_t2.json"
    data = json.load(fr)
    data = sample_list(data, number_=1000)
    for line in data:
      instruction=line["instruction"]
      sentence=line["inputsentence"]
      ground_truth=line["response"]
      predicted=make_inference(instruction,sentence)
      i=i+1
      print(i)
      
      Dic_={}
      Dic_["sentence"]=sentence
      Dic_["ground_truth"]=ground_truth
      Dic_["predicted"]=predicted

      fw.write(json.dumps(Dic_))
      fw.flush()
      fw.write("\n")

  fw.close()
  print(datetime.datetime.now())


if __name__=="__main__":
  args = get_arguments()
  # checkpoints =  ['1000', '2000', '3000', '4000', '5000']
  # args.checkpoints = args.checkpoints.split(',')
  for checkpoint in args.checkpoints.split(','):
    model, tokenizer = load_model_tokenizer_w_saved_lora(args, "checkpoint-%s"%checkpoint)
    save_file_name = args.output_dir + '/' + "test_inference_groundtruth_%s.json"%checkpoint
    gen(args.testset, save_file_name, model, tokenizer)