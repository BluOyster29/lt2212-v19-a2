import argparse, operator, os, numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from nltk import word_tokenize, FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

parser = argparse.ArgumentParser(description='Reads a text and give us analysis')
parser.add_argument("filename", help="takes folder name name as argument", type=str)
parser.add_argument("-B", help="shows top words and counts", type=int)
parser.add_argument("--to-lower", help="processes the text as lower case", action = "store_true")
parser.add_argument("-T", help="counts will be transformed from raw counts into tf-idf", action ="store_true")
parser.add_argument("-S", help="sklearn will transform document vector to dimensionality n", type=int)

def open_text(subfolder, folder_name, file_path, lower):
    '''function takes the name of the subfolder, folder, file_paths for the text files and the bool lower 
    to iterate through text files. The function outputs processed, tokenized text in a list. The tokenized text 
    has all lowercase with the punctuation, numbers and stopwords removed'''

    file = folder_name + "/" + subfolder + '/' + file_path
    tokenizer = RegexpTokenizer(r'[A-Za-z]+[^[0-9]') #regex to remove numbers and punctuation
    tokenised = []
    
    with open(file, "r", encoding="utf-8") as text:
        if lower == True:
            for line in text:
                tokenised += tokenizer.tokenize(line.lower())
        else:
            for line in text:
                tokenised += tokenizer.tokenize(line)
    
    filtered_text = [i for i in tokenised if i not in stopwords.words('english')]
    
    return filtered_text #list of tokens, lower case, minus punctuation and numbers

def text_stats(list_tokens, corpus, top):
    
    '''function takes the processed text, corpus tokens and returns a dictionary with the 
    frequency of the specified tokens, if top is true the function ignores all tokens that 
    are not in the top words corpus''' 

    if top == True:
        
        file_tokens = [i for i in list_tokens if i in corpus.keys()]
        file_freqs = dict(FreqDist(file_tokens))
        return file_freqs #dictionary with top token frequencies
    
    else:
        corpus_freqs = dict(FreqDist(list_tokens))
        return corpus_freqs #dictionary with standard frequencies

def generate_corpus(folder_name, top, n):
    lower = True #change this to argparser 
    subfolders = [i for i in os.listdir(folder_name)]
    corpus_list = []
    for i in subfolders:
        for v in os.listdir(folder_name + "/" + i):
            text = open_text(i,folder_name, v,lower)
            corpus_list += [i for i in text]
    
    
    corpus_freqs = FreqDist(corpus_list)
    sorted_x = sorted(corpus_freqs.items(), key=operator.itemgetter(1), reverse=True)
    if top == True:
        topn_words = {}
        for i in sorted_x[:n]:
            topn_words[i[0]] = 0
        vocabulary = list(sorted(topn_words.keys()))
        return topn_words, vocabulary
            
    else:
        vocabulary = list(sorted(corpus_freqs.keys()))
        corpus = {str(i):0 for i in sorted(vocabulary)}
        return corpus, vocabulary
    '''
    #topn_corpus = text_stats(corpus_list, 10)
    corpus = {str(i):0 for i in sorted(corpus_list)}
    #topn_corpus_t = {str(i):0 for i in topn_corpus}
    vocabulary = list(corpus.keys())
    return 'corpus', vocabulary
    '''
    
def generate_files(folder_name, corpus,top, n):
    lower = True
    subfolders = [i for i in os.listdir(folder_name)]
    files = []
    file_names = []
    for i in subfolders:
        for v in sorted(os.listdir(folder_name + "/" + i)):
            text_vector = text_stats(open_text(i,folder_name,v ,lower),corpus, top)
            corpy = {**corpus, **text_vector}
            #files.append([i + '_' + v, corpy]) file with the name
            files.append(corpy)
            file_names.append(i + '_' + v)

    vect_ar = np.array(files)        
    return vect_ar,file_names
        
    #return files

def generate_dataframe(vectors, file_names, vocabulary):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    df = pd.DataFrame(data=[i for i in vectors], index=[i for i in file_names], columns = [i for i in vocabulary])
    return df

def find_duplicates(data_frame):
    final_matrix = data_frame.drop_duplicates()
    dups_list = data_frame[data_frame.duplicated()].index.tolist()
    
    print("The following indexes have been dropped as duplicates: ")
    for i in dups_list:
        print(i)
    return final_matrix

def generate_tdidf(vector_matrix):
    tfidf_matrix = []
    transformer = TfidfTransformer()
    for i in vector_matrix:
        tfidf_matrix.append(list(i.values()))
    tfidf = transformer.fit_transform(tfidf_matrix).toarray()
    return tfidf

def output_file(data_frame_raw, vocabulary, file_names, top, tdidf, truncated_svd, m):
   
    if top == True and truncated_svd == False and tdidf == False:
        #one file containing a term-document matrix with a vocabulary restriction of your choice, but allowing significantly fewer terms than the total number and no other transformations.
        #python3 gendoc.py -B100 reuters-topics
        data_frame_raw.to_csv(path_or_buf='vectorfiletraw_top.csv',columns=[i for i in vocabulary], index=[i for i in file_names])
    elif tdidf == True and truncated_svd == False:
        #one file containing a term-document matrix with no vocabulary restriction, but with tf-idf applied.
        #python3 gendoc.py -T reuters-topics
        data_frame_raw.to_csv(path_or_buf='vectorfileraw_idf.csv',columns=[i for i in vocabulary], index=[i for i in file_names])
    elif tdidf == True and top == True:
        #one file containing the same vocabulary restriction as in (2), but with tf-idf applied.
        #python3 gendoc.py -B100 -T reuters-topics
        data_frame_raw.to_csv(path_or_buf='vectorfiletdidf_top.csv',columns=[i for i in vocabulary], index=[i for i in file_names])
    elif truncated_svd == True and m == 100 and tdidf == False:
        #one file with no vocabulary restriction, with truncated SVD applied to 100 dimensions.
        #python3 gendoc.py -B100 -T -S100 reuters-topics
        data_frame_raw.to_csv(path_or_buf='vectorfiletruncatedm100.csv',columns=[i for i in vocabulary], index=[i for i in file_names])
    elif truncated_svd == True and m == 1000 and tdidf == False:
        #one file with no vocabulary restriction, with truncated SVD applied to 1000 dimensions.
        #python3 gendoc.py -S1000 reuters-topics
        data_frame_raw.to_csv(path_or_buf='vectorfiletruncated_m1000.csv',columns=[i for i in vocabulary], index=[i for i in file_names])
    elif truncated_svd == True and m == 100 and tdidf == True:
        #one file as in (3), but with truncated SVD applied to 100 dimensions.
        #python3 gendoc.py  -T -S100 reuters-topics
        data_frame_raw.to_csv(path_or_buf='vectorfiletruncatedm100_tdidf.csv',columns=[i for i in vocabulary], index=[i for i in file_names])
    elif truncated_svd == True and m == 1000 and tdidf == True:
        #one file as in (3), but with truncated SVD applied to 1000 dimensions.
        #python3 gendoc.py  -T -S1000 reuters-topics
        data_frame_raw.to_csv(path_or_buf='vectorfiletruncated_m1000_tdidf.csv',columns=[i for i in vocabulary], index=[i for i in file_names])
    else:
        #one file containing a term-document matrix with no vocabulary restriction and no other transformations.
        #python3 gendoc.py reuters-topics
        data_frame_raw.to_csv(path_or_buf='vectorfileraw.csv', index = [i for i in file_names], columns= [i for i in vocabulary])
     
def parsing(parsing):

    args = parsing
    if args.B:
        n = args.B
        top = True
    else:
        n = 10
        top = False
    if args.T: 
        tdidf = True
    else:
        tdidf = False

    if args.S:
        truncated = True
        m = args.S
    else:
        truncated = False
        m = None

    
    return top, n, tdidf, truncated, m , args.filename

def truncated_svd(vectors, m):
    svd = TruncatedSVD(n_components=m)
    data = []
    for i in vectors:
        data.append(list(i.values()))
    svd_data = svd.transform(data)
    return svd_data
    
def main(top, n , tdidf, truncated, m, foldername):
    print("Generating Corpus")
    corpus,vocabulary = generate_corpus(foldername,top, n)
    print("Corpus Generated\nGenerating Vectors")
    vectors,file_names = generate_files(foldername, corpus,top, n)
    print("Vectors Generated")
    
    if tdidf == True and truncated == False:
        print('generating tdidf vectors')
        vectors = generate_tdidf(vectors)
        print('generating tdidf dataframe')
        data_frame = generate_dataframe(vectors,file_names, vocabulary)
    
    if truncated == True and tdidf == True:
        print('generating tdidf vectors')
        vectors = generate_tdidf(vectors)
        print("Truncating")
        data_frame = TruncatedSVD(vectors, m)

    print("Generating Dataframe")
    data_frame = generate_dataframe(vectors,file_names, vocabulary)
    print("Removing Duplicates")
    final_data = find_duplicates(data_frame)
    print("outputting file")
    output_file(final_data, vocabulary, file_names, top, tdidf, truncated, m)
    print("done")
    
if __name__ == "__main__":
    
    top, n, tdidf, truncated, m , foldername = parsing(parser.parse_args())
    main(top, n , tdidf, truncated, m, foldername)

    print(top,n,tdidf,truncated_svd,m,foldername)
    
'''

import goo as g
import pandas as pd
foldername='reuters-topics'
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
lower = True
top = True
n = 20
corpus, vocabulary = g.generate_corpus(foldername,top, n)
vectors, file_names = g.generate_files(foldername, corpus,top, n)
data_frame = g.generate_dataframe(vectors,file_names)
data = g.find_duplicates(data_frame)
g.output_file(data, vocabulary, file_names)

'''
