import itertools
import argparse
import re
from itertools import combinations, groupby, product


parser = argparse.ArgumentParser()     
parser.add_argument('--listname', type=str, default='xxx', help='List number')
args = parser.parse_args()

if args.listname == 'all':
    pre_input_files = [
        "RE_BioRED", "RE_DDI", "RE_git", \
        "EE_genia2011", "EE_phee", "EE_genia2013", \
        "NER_bc2gm", "NER_bc4chemd", "NER_bc5cdr", \
        "TC_ade", "TC_healthadvice", "TC_pubmed20krct"
    ]
    off_set = 0
    output_dir = ""
elif args.listname == 'RE':
    pre_input_files = [
        "RE_BioRED", "RE_DDI", "RE_git"
    ]
    off_set = 20
    output_dir = "RE/"
elif args.listname == 'EE':
    pre_input_files = [
        "EE_genia2011", "EE_phee", "EE_genia2013"
    ]
    off_set = 30
    output_dir = "EE/"
elif args.listname == 'NER':
    pre_input_files = [
        "NER_bc2gm", "NER_bc4chemd", "NER_bc5cdr"
    ]
    off_set = 40
    output_dir = "NER/"
elif args.listname == 'TC':
    pre_input_files = [
        "TC_ade", "TC_healthadvice", "TC_pubmed20krct"
    ]
    off_set = 50
    output_dir = "TC/"
elif args.listname == 'cross1':
    pre_input_files = [
        "RE_BioRED", "EE_genia2011", "NER_bc2gm", "TC_ade"
    ]
    off_set = 100
    output_dir = args.listname + "/"
elif args.listname == 'cross2':
    pre_input_files = [
        "RE_DDI", "EE_phee", "NER_bc4chemd", "TC_healthadvice"
    ]
    off_set = 120
    output_dir = args.listname + "/"
elif args.listname == 'cross3':
    pre_input_files = [
        "RE_git", "EE_genia2013", "NER_bc5cdr", "TC_pubmed20krct"
    ]
    off_set = 140
    output_dir = args.listname + "/"
elif args.listname == 'all_each_group':
    pre_input_files = [
        "RE_BioRED", "RE_DDI", "RE_git", 
        "EE_genia2011", "EE_phee", "EE_genia2013", 
        "NER_bc2gm", "NER_bc4chemd", "NER_bc5cdr", 
        "TC_ade", "TC_healthadvice", "TC_pubmed20krct"
    ]

    # Group the files into chunks of three
    groups = [pre_input_files[i:i + 3] for i in range(0, len(pre_input_files), 3)]

    # Generate all combinations with one file from each group
    all_combinations = list(itertools.product(*groups))

    # Convert each combination tuple to a comma-separated string
    input_files_list = [",".join(comb) for comb in all_combinations]
    off_set = 160
    output_dir = args.listname + "/"
else:
    print('no valid input for list name\n'*10)

# 生成所有组合
# input_files_list = []
# for i in range(2, len(pre_input_files) + 1):
#     for combination in itertools.combinations(pre_input_files, i):
#         input_files_list.append(",".join(combination))




# pre_input_files = [
#     "RE_BioRED", "RE_DDI", "RE_git", \
#     "EE_genia2011", "EE_phee", "EE_genia2013", \
#     "NER_bc2gm", "NER_bc4chemd", "NER_bc5cdr", \
#     "TC_ade", "TC_healthadvice", "TC_pubmed20krct"
# ]

# # Function to extract category
# def get_category(file_name):
#     return re.match(r'^[A-Z]+_', file_name).group(0)

# # Group files by category
# category_dict = {}
# for file in pre_input_files:
#     category = get_category(file)
#     if category not in category_dict:
#         category_dict[category] = []
#     category_dict[category].append(file)

# # Generate all combinations of files from different categories
# input_files_list = []
# for cat1, files1 in category_dict.items():
#     for cat2, files2 in category_dict.items():
#         if cat1 != cat2:
#             for combination in product(files1, files2):
#                 input_files_list.append(combination)



# 

# submit.sh 的模板 dos-chs,
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
# sh {run_script} > result

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
OUTPUT_DIR="saved_models/{output_dir}${{MODEL_NAME##*/}}_${{base_file}}"

echo "Starting training"

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

wait
echo "Starting the generation and metrics scripts"

for FILE in "${{FILES[@]}}"; do
    test_file="${{FILE}}_test"
    OUTPUT_DIR_GENERATION="outputs/{output_dir}${{base_file}}_FOR_${{FILE}}"

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

# 创建.sh文件

for i, input_files in enumerate(input_files_list, 1):
    run_filename = f"run_{i+off_set}.sh"
    submit_filename = f"job_{i+off_set}.sh"
    
    # 生成 run.sh 文件
    # with open(run_filename, 'w') as f:
    #     f.write(run_template.format(input_files=input_files, output_dir=output_dir))
    
    # 生成 submit.sh 文件
    with open(submit_filename, 'w') as f:
        f.write(submit_template.format(input_files=input_files, output_dir=output_dir,run_script=run_filename))

print(f"Generated {len(input_files_list)} pairs of run.sh and submit.sh files.")

















# run_template  = """#!/bin/bash

# # meta-llama/Meta-Llama-3-8B-Instruct
# # "meta-llama/Llama-2-7b-hf"

# # EE_genia2011,EE_phee,NER_bc5cdr

# MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
# INPUT_FILES="{input_files}"
# TRAINING_FILES=$(echo $INPUT_FILES | sed 's/,/ /g' | awk '{{for(i=1;i<=NF;i++) printf("%s_train%s", $i, (i==NF?"":","))}}')

# MAX_STEPS=5000
# SAVE_STEPS=1000
# BATCH_SIZE=1
# GRADIENT_ACCUMULATION_STEPS=4
# LEARNING_RATE=2e-5
# MAX_SEQ_LENGTH=4096

# CHECKPOINTS=$(seq $SAVE_STEPS $SAVE_STEPS $MAX_STEPS | paste -sd, -)
# IFS=',' read -ra FILES <<< "$INPUT_FILES"
# base_file=$(echo $INPUT_FILES | sed -E 's/[^,]*_([^,]*)/\\1/g' | tr ',' '_')
# OUTPUT_DIR="saved_models/{output_dir}${{MODEL_NAME##*/}}_${{base_file}}"

# echo "Starting training"

# # python 0_train.py \\
# #     --sourcefile $TRAINING_FILES \\
# #     --outputdir $OUTPUT_DIR \\
# #     --model_name $MODEL_NAME \\
# #     --max_steps $MAX_STEPS \\
# #     --save_steps $SAVE_STEPS \\
# #     --batch_size $BATCH_SIZE \\
# #     --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \\
# #     --learning_rate $LEARNING_RATE \\
# #     --max_seq_length $MAX_SEQ_LENGTH

# wait
# echo "Starting the generation and metrics scripts"

# for FILE in "${{FILES[@]}}"; do
#     test_file="${{FILE}}_test"
#     OUTPUT_DIR_GENERATION="outputs/{output_dir}${{base_file}}_FOR_${{FILE}}"

#     python 1_generation.py \\
#         --lora_path $OUTPUT_DIR \\
#         --testset $test_file \\
#         --model_name $MODEL_NAME \\
#         --checkpoints $CHECKPOINTS \\
#         --output_dir $OUTPUT_DIR_GENERATION

#     python 2_metrics.py \\
#         --checkpoints $CHECKPOINTS \\
#         --input_dir $OUTPUT_DIR_GENERATION \\
#         --output_dir $OUTPUT_DIR_GENERATION
# done
# """




# submit_template = """#!/bin/bash
# #SBATCH --nodes=1
# #SBATCH -p dos-chs,a100-4,agsmall,ag2tb,a100-8,amdsmall,amdlarge,amd512,amd2tb
# #SBATCH --gres=gpu:a100:1
# #SBATCH --mem=150g
# #SBATCH --time=24:00:00
# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=zhan8023@umn.edu

# conda activate myenv
# module load cuda/12.0
# nvidia-smi
# # sh {run_script} > result
# """