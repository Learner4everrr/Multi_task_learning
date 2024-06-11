from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

auto_list = ['meta-llama/Llama-2-13b-hf', 'medalpaca/medalpaca-13b', 'ncbi/MedCPT-Query-Encoder', 'facebook/contriever',\
							'meta-llama/Meta-Llama-3-8B-Instruct', 'meta-llama/Llama-2-7b-hf']
llama_list = ['chaoyi-wu/MedLLaMA_13B']

def model_creator(*args):
	model_name = args[0]
	config = AutoConfig.from_pretrained(model_name)
	config.max_position_embeddings = 4096
	if len(args) == 1:
		if model_name in auto_list:
			tokenizer = AutoTokenizer.from_pretrained(model_name)
			model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
		elif model_name in llama_list:
			tokenizer = LlamaTokenizer.from_pretrained(model_name)
			model = LlamaForCausalLM.from_pretrained(model_name, config=config)
		else:
			raise ValueError(f"Unknown model name: {model_name}, not in the predefined list")
			
	else:
		bnb_config = args[1]
		if model_id in auto_list:
			tokenizer = AutoTokenizer.from_pretrained(model_id)
			model = AutoModelForCausalLM.from_pretrained(
			    model_id,
			    quantization_config=bnb_config,torch_dtype=torch.float16, device_map='auto', config=config
			)
		elif model_id in llama_list:
			tokenizer = LlamaTokenizer.from_pretrained(model_id)
			model = LlamaForCausalLM.from_pretrained(
			    model_id,
			    quantization_config=bnb_config,torch_dtype=torch.float16, device_map='auto', config=config
			)
		else:
			raise ValueError(f"Unknown model name: {model_name}, not in the predefined list")


	return tokenizer, model



	