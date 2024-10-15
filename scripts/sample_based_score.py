import argparse
import csv
import math
from sentence_transformers import SentenceTransformer

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--input-dir", required=True, help="path to the dir with all inferences files.")
    parser.add_argument("--number-inferences", required=True, help="number of stochastic inference files, excluding the original inference")
    parser.add_argument("--output-path", required=True, help="path to the output file")
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
    # T is number of stochastic inferences. 
    vro = 0
    for i in range(T):
        vro += (1 - similarities[0][i])
    VRO = 1 - vro / T
    return VRO   



if __name__=="__main__":

    args = parse_args()

    print("Input dir is: ", args.input_dir)
    print("Output file is: ", args.output_path)

    all_rows = []
    T = int(args.number_inferences) 
    for i in range(T + 1):
        f = open(args.input_dir + "/inference_{}.csv".format(i))
        reader = csv.reader(f)
        all_rows.append(list(reader))

        if (i>0):
            assert len(all_rows[i]) == len(all_rows[0])
    

    model = SentenceTransformer("all-MiniLM-L6-v2")

    with open(args.output_path, 'w') as file_output:
        file_output.write("data_id,sample_VR,sample_VRO\n")
        for data_id in range(1, len(all_rows[0])):
            p_LM = all_rows[0][data_id][-1]
            p_all = []
            for i in range(1,len(all_rows)):
                p_all.append(all_rows[i][data_id][-1])
          
            if p_LM in p_all or p_all[0] in p_all[1:] or p_all[1] in p_all[2:]:
                print("this data point is not recorded", data_id -1)
                continue
            
            embeddings_all = model.encode(p_all)
            embeddings_LM = model.encode([p_LM])

            similarities_VRO = model.similarity(embeddings_LM, embeddings_all)
            similarities_VR = model.similarity(embeddings_all, embeddings_all)

            VRO = compute_VRO(similarities_VRO, T)
            VR = compute_VR(similarities_VR, T)
            file_output.write(f"{data_id-1},{VR},{VRO}\n")

    file_output.close()



    
