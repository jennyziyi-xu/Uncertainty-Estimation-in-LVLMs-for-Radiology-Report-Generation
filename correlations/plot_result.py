import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 

def parse_args():
    parser = argparse.ArgumentParser(description="Plot LLM metrics vs Uncertainty scores")
    parser.add_argument("--input-metrics", required=True, help="path to input csv file with metrics.")
    parser.add_argument("--input-uq", help="path to input csv file with uq scores.")
    parser.add_argument("--input-sample-uq", help="path to input csv file with sample VR and VRO")
    parser.add_argument("--max-pert-uq", help="path to input csv file with Max perturbation VR and VRO")
    parser.add_argument("--min-pert-uq", help="path to input csv file with Min perturbation VR and VRO")
    parser.add_argument("--max-diff-pert-uq", help="path to input csv file with MaxDiff perturbation VR and VRO")
    args = parser.parse_args()
    return args 


def check_study_subject_id(df_eval, df_uq):
    study_ids = df_eval['study_id'].values
    subject_ids = df_eval['subject_id'].values
    for study_id, subject_id in zip(study_ids, subject_ids):
        if not ((df_uq['study_id']==study_id) & (df_uq['subject_id']==subject_id)).any():
            df_eval = df_eval[~((df_eval['study_id'] == study_id) & (df_eval['subject_id'] == subject_id))]

    study_ids = df_uq['study_id'].values
    subject_ids = df_uq['subject_id'].values
    for study_id, subject_id in zip(study_ids, subject_ids):
        if not ((df_eval['study_id']==study_id) & (df_eval['subject_id']==subject_id)).any():
            df_uq = df_uq[~((df_uq['study_id'] == study_id) & (df_uq['subject_id'] == subject_id))]
    return (df_eval, df_uq)

def check_data_id(df_eval, df_uq):
    data_ids = df_eval['data_id'].values
    for data_id in data_ids:
        if not (df_uq['data_id']==data_id).any():
            print("data_id removed", data_id)
            df_eval = df_eval[~(df_eval['data_id'] == data_id)]

    data_ids = df_uq['data_id'].values
    for data_id in data_ids:
        if not (df_eval['data_id']==data_id).any():
            print("data_id removed", data_id)
            df_uq = df_uq[~(df_uq['data_id'] == data_id)]
    return (df_eval, df_uq)


if __name__ == "__main__":

    args = parse_args()

    # Read the input-metrics file and sort it according to "data_id"
    df_metrics = pd.read_csv(args.input_metrics)
    sorted_df_metrics = df_metrics.sort_values(by='data_id', ignore_index = True)

    all_uq = {}

    all_data_ids  = {}

    # Read the input_uq file
    if args.input_uq:
        print("Adding MaxProb, AvgProb, MaxEntropy, AvgEntropy")
        df_uq = pd.read_csv(args.input_uq)

        sorted_df_metrics, df_uq = check_study_subject_id(sorted_df_metrics, df_uq)

        all_uq["MaxProb"] = df_uq['max_prob']
        all_uq["AvgProb"] = df_uq['avg_prob']
        all_uq["MaxEntropy"] = df_uq['max_entropy']
        all_uq["AvgEntropy"] = df_uq['avg_entropy']
        all_data_ids["uq_data_id"] = df_uq['data_id']

    if args.input_sample_uq:
        df_sample_uq = pd.read_csv(args.input_sample_uq)
        print("Adding SampleVR, SampleVRO")
        sorted_df_metrics, df_sample_uq = check_data_id(sorted_df_metrics, df_sample_uq)

        all_uq["SampleVR"] = df_sample_uq['sample_VR']
        all_uq["SampleVRO"] = df_sample_uq['sample_VRO']
        all_data_ids["sample_data_id"] = df_sample_uq['data_id']
    
    if args.max_pert_uq:
        df_max_pert_uq = pd.read_csv(args.max_pert_uq)
        sorted_df_metrics, df_max_pert_uq = check_study_subject_id(sorted_df_metrics, df_max_pert_uq)
        assert sorted_df_metrics['study_id'].values.tolist() == df_max_pert_uq['study_id'].values.tolist()
        assert sorted_df_metrics['subject_id'].values.tolist() == df_max_pert_uq['subject_id'].values.tolist()
        
        print("Adding MaxPertVR, MaxPertVRO")
        all_uq["MaxPertVR"] = df_max_pert_uq['max_pert_VR']
        all_uq["MaxPertVRO"] = df_max_pert_uq['max_pert_VRO']
    
    if args.min_pert_uq:
        df_min_pert_uq = pd.read_csv(args.min_pert_uq)
        sorted_df_metrics, df_min_pert_uq = check_study_subject_id(sorted_df_metrics, df_min_pert_uq)
        assert sorted_df_metrics['study_id'].values.tolist() == df_min_pert_uq['study_id'].values.tolist()
        assert sorted_df_metrics['subject_id'].values.tolist() == df_min_pert_uq['subject_id'].values.tolist()

        print("Adding MinPertVR, MinPertVRO")
        all_uq["MinPertVR"] = df_min_pert_uq['min_pert_VR']
        all_uq["MinPertVRO"] = df_min_pert_uq['min_pert_VRO']

    if args.max_diff_pert_uq:
        df_max_diff = pd.read_csv(args.max_diff_pert_uq)
        sorted_df_metrics, df_max_diff = check_study_subject_id(sorted_df_metrics, df_max_diff)
        assert sorted_df_metrics['study_id'].values.tolist() == df_max_diff['study_id'].values.tolist()
        assert sorted_df_metrics['subject_id'].values.tolist() == df_max_diff['subject_id'].values.tolist()

        print("Adding MaxDiffVR, MaxDiffVRO")
        all_uq["MaxDiffVR"] = df_max_diff['max_diff_pert_VR']
        all_uq["MaxDiffVRO"] = df_max_diff['max_diff_pert_VRO']

    

    bertscore = sorted_df_metrics['bertscore'].values
    bleu2score = sorted_df_metrics['bleu2_score'].values
    bleu3score = sorted_df_metrics['bleu3_score'].values
    bleu4score = sorted_df_metrics['bleu4_score'].values
    sembscore = sorted_df_metrics['semb_score'].values
    radgraph_combined = sorted_df_metrics['radgraph_combined'].values
    radcliq = sorted_df_metrics['RadCliQ'].values

    
    all_metrics = {"BertScore": bertscore, "Blue2Score": bleu2score, "Blue3Score": bleu3score, "Blue4Score": bleu4score, "sembScore": sembscore, "RadGraph": radgraph_combined, "RadCliQ": radcliq}


    for (x_label, metric) in all_metrics.items():
        print(x_label)
        for (y_label, uq_score) in all_uq.items():
            
            assert len(metric) == len(uq_score)
            metric_new = []
            uq_score_new = []
            for (x,y) in zip(metric, uq_score):
                if y != 0:
                    metric_new.append(x)
                    uq_score_new.append(y)
            assert len(metric_new) == len(uq_score_new)

            print("length of metric_new is", len(metric_new))
            
            # compute pearson correlation coeff.
            correlation_matrix = np.corrcoef(metric_new, uq_score_new)
            pearson_coeff = round(correlation_matrix[0, 1],3)
            
            print(y_label, pearson_coeff)

            plt.figure(figsize=(5, 5))  
            sns.regplot(x = metric_new, y = uq_score_new)
            plt.title("{} vs {}, pearson coeff = {}".format(x_label, y_label, pearson_coeff))
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.savefig("{}_{}.png".format(x_label, y_label))
    