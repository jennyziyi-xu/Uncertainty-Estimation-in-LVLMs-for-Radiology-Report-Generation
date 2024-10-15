import pandas as pd
import csv
import argparse
import ast

import transformers


def parse_args():
    parser = argparse.ArgumentParser(description="Compute the points of perturbation without radgraph. ")
    parser.add_argument("--input-uq", required=True, help="path to the csv file with all the logits.")
    parser.add_argument("--output-file", required=True, help="path to the output csv file")
    args = parser.parse_args()
    return args 

tokenizer = transformers.LlamaTokenizer("/path/")


def find_token_id(token):
    return tokenizer._convert_token_to_id(token)

if __name__ == "__main__":

    args = parse_args()

    input_file = open(args.input_uq, 'r')
    reader = csv.reader(input_file)
    rows = list(reader)

    output_file = open(args.output_file, 'w')

    output_file.write("data_id,max_entropy_pos,max_entropy_token_id,min_entropy_pos,min_entropy_token_id,max_diff_pos,max_diff_id\n")

    for row in rows[1:]:
        
        data_id = row[0]
        print(data_id)
        token_record = ast.literal_eval(row[-1])
        record_dict = {}  
        for item in token_record[4:]:   
            if item[1] not in ["Im", "pression", ":"]:
                record_dict[item[0]] = (item[-1], item[1])
        
        sorted_data  = sorted(record_dict.items(), key=lambda item: item[1][0])
        max_entropy_pos = sorted_data[-1][0]
        min_entropy_pos = sorted_data[0][0]

        max_entropy_token = sorted_data[-1][1][1]
        min_entropy_token = sorted_data[0][1][1]
        
        max_entropy_token_id = find_token_id(max_entropy_token)
        min_entropy_token_id = find_token_id(min_entropy_token)


        diff_dict = {}   
        prev_entropy = token_record[0][-1]
        for item in token_record[5:]:
            entropy_diff = abs(item[-1] - prev_entropy)
            diff_dict[item[0]] = (entropy_diff, item[1])
            prev_entropy = item[-1]
        
        sorted_data = sorted(diff_dict.items(), key=lambda item: item[1][0])

        max_diff_pos = sorted_data[-1][0]   
        max_diff_token = sorted_data[-1][1][1]
        max_diff_id = find_token_id(max_diff_token)
        output_file.write(f"{data_id},{max_entropy_pos},{max_entropy_token_id},{min_entropy_pos},{min_entropy_token_id},{max_diff_pos},{max_diff_id}\n")
    
    output_file.close()
        



