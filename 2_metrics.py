import json
import re
import ast
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--number', type=str, default='5000', help='checkpoint number')
    parser.add_argument('--checkpoints', type=str, default="['1000', '2000', '3000', '4000', '5000']", help='checkpoint list')
    parser.add_argument('--input_dir', type=str, default='Our_model/', help='dir for saving results')
    parser.add_argument('--output_dir', type=str, default='Our_model/', help='dir for saving results')
    parser.add_argument('--save_instruct', type=str, default='Llama2,2,EE...', help='model_name, epoch, training_set')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    return args


def calclulate_f1(statics_dict, prefix=""):
    prec, recall, f1 = 0, 0, 0
    if statics_dict["c"] != 0:
        prec = float(statics_dict["c"] / statics_dict["p"])
        recall = float(statics_dict["c"] / statics_dict["g"])
        f1 = float(prec * recall) / float(prec + recall) * 2
    return {prefix+"-prec": prec, prefix+"-recall": recall, prefix+"-f1": f1}

def com_res_gold(filename):
    # triple extraction
    state_dict = {"p": 0, "c": 0, "g": 0}

    i=-1
    with open(filename, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            line=line.strip()
            line=json.loads(line)
            i=i+1

            P=set()
            gold=set()

            sentence=line["sentence"].lower()
            ground_truth=line["ground_truth"]
            gold_t=ground_truth.split(", ")
            print(gold_t)
            for g in gold_t:
                gold.add(g)

            predictions=line["predicted"].split("\n\n### Response: \n")[1].split("\n")[0]
            predictions_t = predictions.split(", ")
            print(predictions_t)

            state_dict["p"] += len(P)
            state_dict["g"] += len(gold)
            state_dict["c"] += len(P & gold)

    return state_dict

def save_results(args, checkpoint, all_metirc_results):
    save_instruct = args.save_instruct.split(',')
    model_name, epoch, training_set = save_instruct[0], save_instruct[1], save_instruct[2]
    with open(args.output_dir+f'{model_name}-{epoch}-{training_set}.txt', 'a') as file:

        # result_details = f'Model: {model_name}\nepochs/Steps: {epoch}\n' \
        #     + f"Training set list:{training_set}\n"
        #     + f"checkpoint: {checkpoint}\n"
        result_details = f"checkpoint: {checkpoint}\n"
        file.write(result_details)

        file.write(str(all_metirc_results))
        file.write("\n\n")




if __name__=="__main__":
  args = get_arguments()
  args.checkpoints = ast.literal_eval(args.checkpoints)

  for checkpoint in args.checkpoints:
    input_file_name = args.input_dir + "test_inference_groundtruth_%s.json"%checkpoint
    state_dict = com_res_gold(input_file_name)
    all_metirc_results = calclulate_f1(state_dict)
    save_results(args, checkpoint, all_metirc_results)

