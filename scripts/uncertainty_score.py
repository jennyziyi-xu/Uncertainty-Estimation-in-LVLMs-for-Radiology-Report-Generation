import ast
import csv
import math
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--input-logits", required=True, help="path to the csv file with all the logits.")
    parser.add_argument("--output-file", required=True, help="path to the output csv file")
    args = parser.parse_args()
    return args 

def find_value(data, key_to_find):
    for key, value in data:
        if key == key_to_find:
            return value
    return None  # Return None if the key is not found

def find_token(token_index, tokenizer=None):
    return tokenizer._convert_id_to_token(token_index)

def get_scores(input_file, dot_token=29889, tokenizer=None):
    """
    Input:
    input_file: input_csv file 
    
    Return:
    Dictionary containing all information. 
    """
    return_dict = {}
    f_input = open(input_file, 'r')
    reader = csv.reader(f_input)
    rows = list(reader)
    row = rows[0]

    dict_logits = ast.literal_eval(row[0])
    indices = ast.literal_eval(row[1])
    #### Max Prob
    max_prob_para = []
    max_prob_sentence = float("-inf")
    max_prob_token = None
    max_prob_token_pos = None
    max_prob_token_dict = {}

    #### Avg Prob 
    avg_prob_para = []
    avg_prob_sentence = []

    #### Max entropy
    max_entropy_para = []
    max_entropy_sentence = float("-inf")
    max_entropy_token = None
    max_entropy_token_pos = None
    max_entropy_token_dict = {}

    #### Avg Entropy
    avg_entropy_para = []
    avg_entropy_sentence = []

    token_record = []

    for (token_pos, index) in enumerate(indices[1:-1]):
        # get the probability from dict_logits, this is softmax probability s
        prob = find_value(dict_logits[token_pos], index)

        # If the output token isn't from the top 50.
        if prob is None:
            print("Output token isn't from the top_k")
            continue

        # do negative log as shown in the paper. 
        neg_log_prob = -math.log(prob)
        # print("token_pos", token_pos, "prob",neg_log_prob)

        #### Max Prob
        if neg_log_prob > max_prob_sentence:
            max_prob_sentence = neg_log_prob
            max_prob_token = index
            max_prob_token_pos = token_pos

        #### Avg Prob
        avg_prob_sentence.append(neg_log_prob)

        #### Entropy
        all_probs = dict_logits[token_pos]
        entropy_ij = 0
        for (_, token_prob) in all_probs:
            if (token_prob > 0):
                entropy_ij -= token_prob * math.log(token_prob)
    

        #### Max entropy
        if entropy_ij > max_entropy_sentence:
            max_entropy_sentence = entropy_ij
            max_entropy_token = index
            max_entropy_token_pos = token_pos
        
        #### Avg Entropy
        avg_entropy_sentence.append(entropy_ij)

        # index is only added when I started working with radgraph.
        token_record.append((token_pos, find_token(index, tokenizer), index, prob, neg_log_prob, entropy_ij))

        # A dot is represented by 29889. 
        if (index == dot_token):
            #### Max Prob 
            max_prob_token_dict[len(max_prob_para)] = [find_token(max_prob_token, tokenizer), max_prob_token_pos, max_prob_sentence]
            max_prob_para.append(max_prob_sentence)
            max_prob_sentence = float("-inf")
            max_prob_token = None

            #### Avg Prob
            avg_prob_sentence_score = sum(avg_prob_sentence)/len(avg_prob_sentence)
            avg_prob_para.append(avg_prob_sentence_score)
            avg_prob_sentence = []

            #### Max entropy
            max_entropy_token_dict[len(max_entropy_para)] = [find_token(max_entropy_token, tokenizer), max_entropy_token_pos, max_entropy_sentence]
            max_entropy_para.append(max_entropy_sentence)
            max_entropy_sentence = float("-inf")
            max_entropy_token = None

            #### Avg entropy
            avg_entropy_sentence_score = sum(avg_entropy_sentence)/len(avg_entropy_sentence)
            avg_entropy_para.append(avg_entropy_sentence_score)
            avg_entropy_sentence = []

    ### Max Prob: Take the average across the sentences. 
    score_max_prob = sum(max_prob_para) / len(max_prob_para) if max_prob_para else 0

    ### Avg Prob 
    score_avg_prob = sum(avg_prob_para) / len(avg_prob_para) if avg_prob_para else 0

    ### Max entropy 
    score_max_entropy = sum(max_entropy_para) / len(max_entropy_para) if max_entropy_para else 0

    ### Avg entropy
    score_avg_entropy = sum(avg_entropy_para) / len(avg_entropy_para) if avg_entropy_para else 0 

    return [score_max_prob, max_prob_para, max_prob_token_dict, score_avg_prob, avg_prob_para, score_max_entropy, max_entropy_para, max_entropy_token_dict, score_avg_entropy, avg_entropy_para, token_record]

            



