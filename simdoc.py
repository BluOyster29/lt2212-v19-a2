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
'''
def cos_sametopic(splat_data):
    cosim_list = []
    for i in splat_data:
        if all(v == 0 for v in [num for num in i[1:]]):
            continue
        else:
            cos_sim = dot(splat_data[0][1:], i[1:])/(norm(splat_data[0][1:])*norm(i[1:]))
            cosim_list.append(cos_sim)
    
    return sum(cosim_list) / len(cosim_list) * 100
'''
def cos_sametopic(matrix):
    
    vector1 = [i[1:] for i in matrix]
    return cosine_similarity(vector1,vector1).mean() * 100

def cosine_different_topics(grain_matrix, crude_matrix):

    vector1 = [i[1:] for i in grain_matrix]
    vector2 = [i[1:] for i in crude_matrix]
    return cosine_similarity(vector1,vector2).mean() * 100
  
def main(file):
    #args = parser.parse_args()
    #data = import_dataframe(args.vectorfile)
    data = import_dataframe(file)
    grain_matrix,crude_matrix = split_data(data)
    grain_cosim_average = cos_sametopic(grain_matrix)
    print(file)
    print('\nThe grain files were all {}  similar to themselves'.format(grain_cosim_average))
    crude_cosim_average = cos_sametopic(crude_matrix)
    print('The crude files were all {}  similar to themselves'.format(crude_cosim_average))
    crude_average_sim_different = cosine_different_topics(crude_matrix, grain_matrix)
    print('The topics were all {}  similar to eachother\n'.format(crude_average_sim_different))
   
if __name__ == '__main__':
    files = ['vectorfile_top20.csv','vectorfile_top100.csv','vectorfile_truncated_100.csv',
    'vectorfileraw_tidf.csv', 'vectorfileraw_tdidf_top.csv', 'vectorfileraw.csv',
    'vectorfiletruncated_m1000_tdidf.csv', 'vectorfiletruncated_m1000.csv', 'vectorfiletruncatedm100_tdidf.csv']
    
    for file in files:
        main(file)
   
