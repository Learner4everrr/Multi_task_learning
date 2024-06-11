#!/bin/bash

# python 0_train.py \
# 	--sourcefile EE_genia2011_train,EE_phee_train\
# 	--outputdir saved_models/test/ \
# 	--model_name meta-llama/Llama-2-7b-hf\
# 	--max_steps 2000 \
# 	--save_steps 1000 \
# 	--batch_size 1 \
# 	--gradient_accumulation_steps 16 \
# 	--learning_rate 2e-5 \
# 	--max_seq_length 2048

python 1_generation.py \
	--lora_path saved_models/test \
	--testset EE_genia2011_test.json \
	--model_name meta-llama/Llama-2-7b-hf \
	--checkpoints ['25', '50'] \
	--output_dir outputs/Llama2test