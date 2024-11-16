import os
import pandas as pd
import csv
import sys
import ast
import json

from radgraph_evaluate_model import generate_radgraph



radgraph_model_checkpoint = "/path/"
inference_file = "/path/"
uq_scores_file = "/path/"
new_uq_file = "/path/"


if __name__ == "__main__":
    df_scores = pd.read_csv(uq_scores_file)
    df_inference=pd.read_csv(inference_file)

    f_uq = open(new_uq_file, 'a')
    writer_uq = csv.writer(f_uq, quoting = csv.QUOTE_MINIMAL)
    for index, row in df_scores.iterrows():
        data_id = int(row['data_id'])
        if data_id <= 894:
            continue
        
        study_id = row['study_id']
        subject_id = row['subject_id']
        token_record = ast.literal_eval(row['token_record'])
        row_output = df_inference.iloc[index]
        text_output = row_output['target']
        text_output = text_output.replace("Findings:", '')
        text_output = text_output.replace("Impression:", '')

        text_csv = os.path.join("./output_text.csv")
        f_text = open(text_csv, 'w')
        writer_text = csv.writer(f_text, quoting=csv.QUOTE_MINIMAL)
        f_text.write("data_id,subject_id,study_id,report\n")
        row = [data_id, subject_id, study_id, text_output]
        writer_text.writerow(row)
        f_text.flush()
        f_text.close()

    
        pred_path = "./pred_cache.json"
        generate_radgraph(radgraph_model_checkpoint, text_csv, pred_path, image=False)

        new_token_record = []
        
        with open('./temp_dygie_input.json', 'r') as file:
            data_words = json.load(file)
            words = data_words["sentences"][0]
        
        curr_token_index = 3   
        curr_word = ""
        mapping = {}
        
        curr_word_pos = 0
        curr_target = words[0]
        

        while curr_token_index < len(token_record):
            curr_target = words[curr_word_pos]
            token = token_record[curr_token_index]
            
            token=token[1].lstrip('\u2581')
            curr_word = curr_word + token
            
            
            if (curr_word == "Impression:"):
                del mapping[curr_token_index-1]
                del mapping[curr_token_index-2]
                curr_word = ""
                curr_token_index += 1
                continue
            
            mapping[curr_token_index] = curr_word_pos
            if curr_word == curr_target.strip():
                curr_word_pos +=1
                curr_word = ""
            curr_token_index += 1
                
        with open(pred_path, 'r') as file:
            data_entities = json.load(file)

        retain_idx = []

        entities = list(data_entities.values())[0]["entities"]
        for key, value in entities.items():
            start_idx = value["start_ix"]
            end_idx = value["end_ix"]
            retain_idx.extend(list(range(start_idx, end_idx+1)))
        retain_idx = list(set(retain_idx))
        retain_idx.sort()

        #### Max Prob
        max_prob_para = []
        max_prob_sentence = float("-inf")
        max_prob_token = None
        max_prob_token_idx = None
        max_prob_token_pos = None
        max_prob_token_dict = {}

        #### Avg Prob 
        avg_prob_para = []
        avg_prob_sentence = []

        #### Max entropy
        max_entropy_para = []
        max_entropy_sentence = float("-inf")
        max_entropy_token = None
        max_entropy_token_idx = None
        max_entropy_token_pos = None
        max_entropy_token_dict = {}

        #### Avg Entropy
        avg_entropy_para = []
        avg_entropy_sentence = []

        for token in token_record:
            token_pos = token[0]
            token_index = token[2]
            neg_log_prob = token[-2]
            entropy_ij = token[-1]
            if token_index != 29871 and token_pos in mapping and mapping[token_pos] in retain_idx:

                ## Max Prob
                if neg_log_prob > max_prob_sentence:
                    max_prob_sentence = neg_log_prob
                    max_prob_token_idx = token_index
                    max_prob_token_pos = token_pos
                    max_prob_token = token[1]

                ## Avg Prob
                avg_prob_sentence.append(neg_log_prob)

                ## Max entropy
                if entropy_ij > max_entropy_sentence:
                    max_entropy_sentence = entropy_ij
                    max_entropy_token_idx = token_index
                    max_entropy_token_pos = token_pos
                    max_entropy_token = token[1]

                ## Avg Entropy
                avg_entropy_sentence.append(entropy_ij) 

        
            ## A dot is represented by 29889 followed by a space
            
            if (token_index == 29889 and (((token_pos+1) >= len(token_record)) or token_record[token_pos+1][2] == 29871)):
                ## Max Prob
                if max_prob_sentence != float("-inf"):
                    max_prob_token_dict[len(max_prob_para)]= [max_prob_token, max_prob_token_idx, max_prob_token_pos, max_prob_sentence]
                    max_prob_para.append(max_prob_sentence)
                    max_prob_sentence = float("-inf")
                max_prob_token = None
                max_prob_token_idx = None
                max_prob_token_pos = None

                ## Avg Prob 
                if (avg_prob_sentence):
                    avg_prob_sentence_score = sum(avg_prob_sentence)/len(avg_prob_sentence)
                    avg_prob_para.append(avg_prob_sentence_score)
                avg_prob_sentence = []

                ## Max entropy 
                if max_entropy_sentence != float("-inf"):
                    max_entropy_token_dict[len(max_entropy_para)] = [max_entropy_token, max_entropy_token_idx, max_entropy_token_pos, max_entropy_sentence]
                    max_entropy_para.append(max_entropy_sentence)
                    max_entropy_sentence = float("-inf")
                max_entropy_token = None
                max_entropy_token_idx = None
                max_entropy_token_pos = None

                ## Avg entropy 
                if (avg_entropy_sentence):
                    avg_entropy_sentence_score = sum(avg_entropy_sentence)/len(avg_entropy_sentence)
                    avg_entropy_para.append(avg_entropy_sentence_score)
                avg_entropy_sentence = []
        
        ## Max Prob : take the average across all sentences
        if max_prob_para:   
            score_max_prob = sum(max_prob_para)/len(max_prob_para)
        else:
            score_max_prob = 0

            ## Avg Prob 
        if avg_prob_para:
            score_avg_prob = sum(avg_prob_para)/len(avg_prob_para)
        else:
            score_avg_prob = 0	

            ## Max entropy
        if max_entropy_para:
            score_max_entropy = sum(max_entropy_para) / len(max_entropy_para)
        else:
            score_max_entropy = 0

            ## Avg entropy 
        if avg_entropy_para:        
            score_avg_entropy = sum(avg_entropy_para) / len(avg_entropy_para)
        else:
            score_avg_entropy = 0

        write_result = [data_id, study_id, subject_id, score_max_prob, max_prob_para, max_prob_token_dict, score_avg_prob, avg_prob_para, score_max_entropy, max_entropy_para, max_entropy_token_dict, score_avg_entropy, avg_entropy_para,entities]

        writer_uq.writerow(write_result)

        f_uq.flush()
        
    f_uq.close()
    

