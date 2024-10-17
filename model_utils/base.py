import numpy as np
import torch
import csv
import pandas as pd
import ast
import os



def write_to_logits(logits_file, logits):
    curr_token_pos = None
    np.set_printoptions(threshold=np.inf)
    m = torch.nn.Softmax(dim=0)
    
    if logits.shape[1] != 1:
        logits_array = logits[0][-1]
    else:
        logits_array = logits[0][0]

    logits_softmax = m(logits_array)
    logits_np = logits_softmax.cpu().numpy()
    logits_list = list(enumerate(logits_np))
    logits_sorted = sorted(logits_list, key=lambda x:x[1], reverse=True)
    logits_top_50 = logits_sorted[:50]

    if not os.path.exists(logits_file):
        with open(logits_file, 'w') as file:
            all_logits = {0:logits_top_50}
            file.write(f"\"{all_logits}\",")
            curr_token_pos = 0
    else:
        csv.field_size_limit(100000000)
        with open(logits_file, 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
        last_row = rows[-1]
        if len(list(last_row[1])) != 0 :

            all_logits = {0:logits_top_50}
            curr_token_pos = 0

            with open(logits_file, 'w') as file:
                file.write(f"\"{all_logits}\",")
        else:
            dict_logits = ast.literal_eval(last_row[0])
    
            last_index = max(dict_logits.keys())
            dict_logits[last_index + 1] = logits_top_50
            with open(logits_file, 'w', newline='') as file:
                file.write(f"\"{dict_logits}\",")
            curr_token_pos = last_index + 1
    return curr_token_pos


def write_output_tokens(logits_file, outputs):
    np.set_printoptions(threshold=np.inf)
    outputs0_cpu = outputs.cpu()
    outputs0_np = outputs0_cpu.numpy()

    csv.field_size_limit(100000000)
    with open(logits_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    last_row = rows[-1]
    last_row_logits = ast.literal_eval(last_row[0])

    with open(logits_file, 'w', newline='') as file:
        file.write(f"\"{last_row_logits}\",\"{list(outputs0_np)}\"\n")
        file.flush()
    file.close()
    return


def perturbation(flag_type, points_perturb_file, inference_file, curr_token_pos, inferece_number, logits):
    assert (flag_type in ["max", "min", "max_diff"])
    assert curr_token_pos is not None
    token_pos = None
    token_id = None

    df_perturb = pd.read_csv(points_perturb_file)
    valid_data_ids = df_perturb['data_id'].values

    df_inference = pd.read_csv(inference_file)
    num_inferences = len(df_inference)
    
    if (num_inferences in valid_data_ids):
        row_perturb = df_perturb[df_perturb['data_id'] == num_inferences]

        if (flag_type == "max"):
            token_pos = row_perturb['max_entropy_pos'].values[0]
            token_id = row_perturb['max_entropy_token_id'].values[0]
        elif (flag_type == "min"):
            token_pos = row_perturb['min_entropy_pos'].values[0]
            token_id = row_perturb['min_entropy_token_id'].values[0]
        elif (flag_type == "max_diff"):
            token_pos = row_perturb['max_diff_pos'].values[0]
            token_id = row_perturb['max_diff_id'].values[0]
        

    else:
        print("{} is invalid.".format(num_inferences))    

    if (curr_token_pos == token_pos):
        if logits.shape[1] != 1:
            logits[0][-1][token_id] = float("-inf")

            if inferece_number >= 2:
                
                max_id = logits[0][-1].argmax()
                logits[0][-1][max_id] = float("-inf")

            if inferece_number == 3:
                
                max_id = logits[0][-1].argmax()
                logits[0][-1][max_id] = float("-inf")

        else:
            logits[0][0][token_id] = float("-inf")

            if inferece_number >= 2:
                
                max_id = logits[0][0].argmax()
                logits[0][0][max_id] = float("-inf")

            if inferece_number == 3:
                
                max_id = logits[0][0].argmax()
                logits[0][0][max_id] = float("-inf")
