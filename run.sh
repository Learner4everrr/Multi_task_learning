#!/bin/bash

python 0_train.py \
	--sourcefile EE_genia2011_train,EE_phee_train\
	--outputdir saved_models/test/ \
	--model_name meta-llama/Llama-2-7b-hf\
	--max_steps 50 \
	--save_steps 25 \
	--batch_size 8\
	--gradient_accumulation_steps 4\
	--learning_rate 2e-5