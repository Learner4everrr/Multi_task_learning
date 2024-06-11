#!/bin/bash

python 0_train.py \
	--sourcefile EE_genia2011_train,EE_phee_train\
	--outputdir saved_models/test/ \
	--model_name meta-llama/Llama-2-7b-hf\
	--max_steps 100 \
	--save_steps 20 \
	--batch_size 2 \
	--gradient_accumulation_steps 16 \
	--learning_rate 2e-5 \
	--max_seq_length 2048