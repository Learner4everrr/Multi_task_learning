#!/bin/bash

# meta-llama/Meta-Llama-3-8B-Instruct
# "meta-llama/Llama-2-7b-hf"


MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
INPUT_FILES="EE_genia2011,EE_phee,NER_bc5cdr"
TRAINING_FILES=$(echo $INPUT_FILES | sed 's/,/ /g' | awk '{for(i=1;i<=NF;i++) printf("%s_train%s", $i, (i==NF?"":","))}')

MAX_STEPS=1000
SAVE_STEPS=500
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=16
LEARNING_RATE=2e-5
MAX_SEQ_LENGTH=4096

CHECKPOINTS=$(seq $SAVE_STEPS $SAVE_STEPS $MAX_STEPS | paste -sd, -)
IFS=',' read -ra FILES <<< "$INPUT_FILES"
base_file=$(echo $INPUT_FILES | sed -E 's/[^,]*_([^,]*)/\1/g' | tr ',' '_')
OUTPUT_DIR="saved_models/${MODEL_NAME##*/}_${base_file}"

python 0_train.py \
    --sourcefile $TRAINING_FILES \
    --outputdir $OUTPUT_DIR \
    --model_name $MODEL_NAME \
    --max_steps $MAX_STEPS \
    --save_steps $SAVE_STEPS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --max_seq_length $MAX_SEQ_LENGTH

# printf "%.0s1" {1..100}

for FILE in "${FILES[@]}"; do
	test_file="${FILE}_test"
    OUTPUT_DIR_GENERATION="outputs/${base_file}_FOR_${FILE}"

	python 1_generation.py \
	    --lora_path $OUTPUT_DIR \
	    --testset $test_file \
	    --model_name $MODEL_NAME \
	    --checkpoints $CHECKPOINTS \
	    --output_dir $OUTPUT_DIR_GENERATION

	python 2_metrics.py \
    --checkpoints $CHECKPOINTS \
    --input_dir $OUTPUT_DIR_GENERATION \
    --output_dir $OUTPUT_DIR_GENERATION \
    # --save_instruct $SAVE_INSTRUCT
    # printf "%.0s2" {1..100}
done



