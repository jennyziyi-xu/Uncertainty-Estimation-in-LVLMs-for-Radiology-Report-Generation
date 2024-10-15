import argparse
import csv
import math
from sentence_transformers import SentenceTransformer
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Given a directory of 3 different inferences and the original inference, compute perturbation VR and VRO scores. ")
    parser.add_argument("--inference-0", required=True, help="path to inference_0.csv")
    parser.add_argument("--input-dir", required=True, help="path to the dir with all inferences files.")
    parser.add_argument("--number-inferences", required=True, help="number of stochastic inference files, excluding the original inference")
    parser.add_argument("--output-path", required=True, help="path to the output file")
    parser.add_argument("--flag-type", required=True, help="must be one of 'max', 'min', 'maxDiff'")
    args = parser.parse_args()
    return args 


def compute_VR(similarities, T):
    outer_sum = 0
    for i in range(T):
        inner_sum = 0
        for j in range(T):
            if (i != j):
                inner_sum += (1 - similarities[i][j])
        outer_sum += inner_sum / (T-1)
    VR = 1 - outer_sum / T
    return VR

def compute_VRO(similarities, T):
    vro = 0
    for i in range(T):
        vro += (1 - similarities[0][i])
    VRO = 1 - vro / T
    return VRO   


def check_study_subject_id(study_id, subject_id, f1, f2, f3):
    result = True
    for df in [f1, f2, f3]:
        if not ((df['study_id']==study_id) & (df['subject_id']==subject_id)).any():
            print("check_study_subject_id returns False")
            result = False
    return result


if __name__=="__main__":

    args = parse_args()
    T=int(args.number_inferences)

    print("Input dir is: ", args.input_dir)
    print("Output file is: ", args.output_path)

    assert args.flag_type in ["max", "min", "maxDiff"]

    all_rows = []

    f_0 = pd.read_csv(args.inference_0)

    f_1 = pd.read_csv(args.input_dir + "/inference_1.csv")

    f_2 = pd.read_csv(args.input_dir + "/inference_2.csv")

    f_3 = pd.read_csv(args.input_dir + "/inference_3.csv")

    study_ids = f_0['study_id'].values
    subject_ids = f_0['subject_id'].values

    for study_id, subject_id in zip(study_ids, subject_ids):
        contains_value = check_study_subject_id(study_id, subject_id, f_1, f_2, f_3)
        if not contains_value:
            f_0 = f_0[~((f_0['study_id'] == study_id) & (f_0['subject_id'] == subject_id))]
            f_1 = f_1[~((f_1['study_id'] == study_id) & (f_1['subject_id'] == subject_id))]
            f_2 = f_2[~((f_2['study_id'] == study_id) & (f_2['subject_id'] == subject_id))]
            f_3 = f_3[~((f_3['study_id'] == study_id) & (f_3['subject_id'] == subject_id))]
    
    assert len(f_0) == len(f_1) == len(f_2) == len(f_3)


    model = SentenceTransformer("all-MiniLM-L6-v2")

    with open(args.output_path, 'w') as file_output:

        if (args.flag_type == "max"):
            file_output.write("subject_id,study_id,max_pert_VR,max_pert_VRO\n")

        if (args.flag_type == "min"):
            file_output.write("subject_id,study_id,min_pert_VR,min_pert_VRO\n")
        
        if (args.flag_type == "maxDiff"):
            file_output.write("subject_id,study_id,max_diff_pert_VR,max_diff_pert_VRO\n")

        for i in range(len(f_0)):
            subject_id = f_0.iloc[i]['subject_id']
            study_id = f_0.iloc[i]['study_id']
            for df in [f_1, f_2, f_3]:
                assert(df.iloc[i]['subject_id'] == subject_id)
                assert(df.iloc[i]['study_id'] == study_id)
            p_LM = f_0.iloc[i]['target']

            p_all = [f_1.iloc[i]['target'], f_2.iloc[i]['target'],f_3.iloc[i]['target']]
          
            if (p_LM in p_all) or (p_all[0] in p_all[1:]) or (p_all[1] in p_all[2:]):
                print("skipping because at least two outputs are matching.")
                print("subject_id:", subject_id)
                print("study_id:", study_id)
                continue
                
            
            embeddings_all = model.encode(p_all)
            embeddings_LM = model.encode([p_LM])

            similarities_VRO = model.similarity(embeddings_LM, embeddings_all)
            similarities_VR = model.similarity(embeddings_all, embeddings_all)

            VRO = compute_VRO(similarities_VRO, T)
            VR = compute_VR(similarities_VR, T)

            file_output.write(f"{subject_id},{study_id},{VR},{VRO}\n")

    file_output.close()



    
