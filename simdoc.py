import argparse, operator, os, numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from numpy import dot
from numpy.linalg import norm

parser = argparse.ArgumentParser(description='Reads a text and give us analysis')
parser.add_argument("vectorfile", help="takes folder name name as argument", type=str)

def import_dataframe(file_path):
    data_frame = pd.read_csv(file_path) 
    data = data_frame.values
    
    return data

def split_data(nparray):
    grain_matrix = []
    crude_matrix = []
    for i in nparray:
        
        if 'grain' in i[0]:
            grain_matrix.append(i)
        else:
            crude_matrix.append(i)

    return grain_matrix, crude_matrix

def cos_sametopic(splat_data):
    cosim_list = []
    for i in splat_data:
        if all(v == 0 for v in [num for num in i[1:]]):
            continue
        else:
            cos_sim = dot(splat_data[0][1:], i[1:])/(norm(splat_data[0][1:])*norm(i[1:]))
            cosim_list.append(cos_sim)
    
    return sum(cosim_list) / len(cosim_list) * 100
    
def cos_different(grain_matrix, crude_matrix, average_file):
    counter = 0 
    cosim_list = []
    for i in grain_matrix:
        for i in crude_matrix:
            if all(v == 0 for v in [num for num in i[1:]]):
                continue
            else:
                cos_sim = dot(grain_matrix[counter][1:], i[1:])/(norm(grain_matrix[counter][1:])*norm(i[1:]))
                cosim_list.append(cos_sim)

        if counter > len(grain_matrix):
            break
        else:
            counter +=1 
    return ((sum(cosim_list) / len(cosim_list)) * 100) / average_file

def main(file):
    args = parser.parse_args()
    
    #data = import_dataframe(args.vectorfile)
    data = import_dataframe(args.vectorfile)
    grain_matrix,crude_matrix = split_data(data)
    #print(grain_matrix)
    grain_cosim_average = cos_sametopic(grain_matrix)
    print(grain_cosim_average)
    crude_cosim_average = cos_sametopic(crude_matrix)
    print(crude_cosim_average)
    crude_average_sim_different = cos_different(crude_matrix, grain_matrix, crude_cosim_average)
    print(crude_average_sim_different)
    grain_average_sim_different = cos_different(grain_matrix, crude_matrix, grain_cosim_average)
    print(crude_average_sim_different)

if __name__ == '__main__':
    files = ['vectorfileraw_top.csv','vectorfileraw_idf.csv','vectorfiletdidf_top.csv',
            'vectorfiletruncatedm100.csv', 'vectorfiletruncated_m1000.csv', 'vectorfiletruncatedm100_tdidf.csv',
            'vectorfiletruncated_m1000_tdidf.csv']
    
    for file in files:
        main(file)
    '''
    args = parser.parse_args()

    data = import_dataframe(args.vectorfile)
    grain_matrix,crude_matrix = split_data(data)
    #print(grain_matrix)
    grain_cosim_average = cos_sametopic(grain_matrix)
    print(grain_cosim_average)
    crude_cosim_average = cos_sametopic(crude_matrix)
    print(crude_cosim_average)
    crude_average_sim_different = cos_different(crude_matrix, grain_matrix, crude_cosim_average)
    print(crude_average_sim_different)
    grain_average_sim_different = cos_different(grain_matrix, crude_matrix, grain_cosim_average)
    print(crude_average_sim_different)
    '''
    #total = cos_different(grain_matrix, crude_matrix)
    #print(total)
    '''
    crude_mean = cosine_similarity([i[1:] for i in crude_matrix])
    grain_mean = cosine_similarity([i[1:] for i in grain_matrix])
    print(crude_mean.mean(), grain_mean.mean())
    '''
    
    
    
    


'''
import simdoc as sim
from sklearn.metrics.pairwise import cosine_similarity
data = sim.import_dataframe('vectorfile.csv')
grain_matrix,crude_matrix = sim.split_data(data)
grain_cosim = sim.cos_sametopic(grain_matrix)

print(array)
grain_matrix,crude_matrix = sim.split_data(array)
crude_mean = sim.cosine_similarity([i[1:] for i in crude_matrix])
grain_ar = sim.cosine_similarity([i[1:] for i in grain_matrix])

'''
