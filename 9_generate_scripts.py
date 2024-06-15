import itertools

pre_input_files = [
    "RE_BioRED", "RE_DDI", "RE_git", \
    "EE_genia2011", "EE_phee", "EE_genia2013", \
    "NER_bc2gm", "NER_bc4chemd", "NER_bc5cdr", \
    "TC_ade", "TC_healthadvice", "TC_pubmed20krct"
]

pre_input_files = [
    "RE_BioRED", "RE_DDI", "RE_git"
]

# 生成所有组合
input_files_list = []
for i in range(1, len(pre_input_files) + 1):
    for combination in itertools.combinations(pre_input_files, i):
        input_files_list.append(",".join(combination))





run_template  = """#!/bin/bash

# meta-llama/Meta-Llama-3-8B-Instruct
# "meta-llama/Llama-2-7b-hf"

# EE_genia2011,EE_phee,NER_bc5cdr

MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
INPUT_FILES="{input_files}"
TRAINING_FILES=$(echo $INPUT_FILES | sed 's/,/ /g' | awk '{{for(i=1;i<=NF;i++) printf("%s_train%s", $i, (i==NF?"":","))}}')

MAX_STEPS=5000
SAVE_STEPS=1000
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=2e-5
MAX_SEQ_LENGTH=4096

CHECKPOINTS=$(seq $SAVE_STEPS $SAVE_STEPS $MAX_STEPS | paste -sd, -)
IFS=',' read -ra FILES <<< "$INPUT_FILES"
base_file=$(echo $INPUT_FILES | sed -E 's/[^,]*_([^,]*)/\\1/g' | tr ',' '_')
OUTPUT_DIR="saved_models/${{MODEL_NAME##*/}}_${{base_file}}"

python 0_train.py \\
    --sourcefile $TRAINING_FILES \\
    --outputdir $OUTPUT_DIR \\
    --model_name $MODEL_NAME \\
    --max_steps $MAX_STEPS \\
    --save_steps $SAVE_STEPS \\
    --batch_size $BATCH_SIZE \\
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \\
    --learning_rate $LEARNING_RATE \\
    --max_seq_length $MAX_SEQ_LENGTH

for FILE in "${{FILES[@]}}"; do
    test_file="${{FILE}}_test"
    OUTPUT_DIR_GENERATION="outputs/${{base_file}}_FOR_${{FILE}}"

    python 1_generation.py \\
        --lora_path $OUTPUT_DIR \\
        --testset $test_file \\
        --model_name $MODEL_NAME \\
        --checkpoints $CHECKPOINTS \\
        --output_dir $OUTPUT_DIR_GENERATION

    python 2_metrics.py \\
        --checkpoints $CHECKPOINTS \\
        --input_dir $OUTPUT_DIR_GENERATION \\
        --output_dir $OUTPUT_DIR_GENERATION
done
"""

# submit.sh 的模板
submit_template = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p dos-chs,a100-4,agsmall,ag2tb,a100-8,amdsmall,amdlarge,amd512,amd2tb
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=150g
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhan8023@umn.edu

conda activate myenv
module load cuda/12.0
nvidia-smi
bash {run_script} > result
"""

# 创建.sh文件
off_set = 20
for i, input_files in enumerate(input_files_list, 1):
    run_filename = f"run_{i+off_set}.sh"
    submit_filename = f"job_{i+off_set}.sh"
    
    # 生成 run.sh 文件
    with open(run_filename, 'w') as f:
        f.write(run_template.format(input_files=input_files))
    
    # 生成 submit.sh 文件
    with open(submit_filename, 'w') as f:
        f.write(submit_template.format(run_script=run_filename))

print(f"Generated {len(input_files_list)} pairs of run.sh and submit.sh files.")