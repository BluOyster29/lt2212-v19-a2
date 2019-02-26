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
    
    #function imports a csv file from disk
    data_frame = pd.read_csv(file_path) 
    data = data_frame.values
    
    return data#returns matrix

def split_data(nparray):

    '''function splits the grain and crude files into independ matrices
    to be passed to the averaging function'''

    grain_matrix = []
    crude_matrix = []
    for i in nparray:
        
        if 'grain' in i[0]: #splitting on whether or not grain is in filename
            grain_matrix.append(i)
        else:
            crude_matrix.append(i)

    return grain_matrix, crude_matrix

def cos_sametopic(splat_data):
    
    '''function for finding cosine similarity of a matrix. Iterates 
    through each vector pair and finds the cosine similarity. Average cosime similarity
    is then calculated and presented'''

    cosim_list = []
    for i in splat_data:
        if all(v == 0 for v in [num for num in i[1:]]): #loop that ignores 0 count vectors
            continue
        else:
            data1 = i[1:]
            data2 = splat_data[0][1:]
            cos_sim = cosine_similarity([data1], [data2])
            cosim_list.append(cos_sim[0][0])
    
    return round((sum(cosim_list) / len(cosim_list)) ,3)

def cosine_different_topics(grain_matrix, crude_matrix):
    
    '''function for calculating cosine similarity between each topic. Function
    makes use of cosine function. Average is calculated then presented'''

    vector1 = [i[1:] for i in grain_matrix]
    vector2 = [i[1:] for i in crude_matrix]
    return round(cosine_similarity(vector1,vector2).mean(), 3)

def main():
    '''main function that calls the other function. First arguments are loaded 
    then data generated followed by the average calculations'''
    args = parser.parse_args()
    data = import_dataframe(args.vectorfile)
    grain_matrix,crude_matrix = split_data(data)
    grain_cosim_average = cos_sametopic(grain_matrix)
    crude_cosim_average = cos_sametopic(crude_matrix)
    crude_average_sim_different = cosine_different_topics(crude_matrix, grain_matrix)
    print(args.vectorfile)
    print('\nThe grain files were on average {}  similar to themselves'.format(grain_cosim_average))
    print('The crude files were on average {}  similar to themselves'.format(crude_cosim_average))
    print('The topics were on average {}  similar to eachother\n'.format(crude_average_sim_different))
   
if __name__ == '__main__':
    #this is where the magic happens
    main()
