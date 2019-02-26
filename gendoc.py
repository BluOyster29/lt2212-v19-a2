
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
    
    '''corpus of words generated to be used as the vocabulary. Function takes into account topn and will 
    create a corpus with the topn amount of tokens if topn is True.'''

    lower = True #activates lowercase tokens
    subfolders = [i for i in os.listdir(folder_name)] #iterates through subfolder 
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
        return topn_words, vocabulary #empty  topn dictionary to be used to populate vectors and vocabulary for columns
            
    else:
        vocabulary = list(sorted(corpus_freqs.keys()))
        corpus = {str(i):0 for i in sorted(vocabulary)}
        return corpus, vocabulary
      
def generate_files(folder_name, corpus,top, n):
    
    '''Function tokenises the text files and adds filenames to the index'''

    lower = True
    subfolders = [i for i in os.listdir(folder_name)]
    files = []
    file_names = []
    for i in subfolders:
        for v in sorted(os.listdir(folder_name + "/" + i)):
            text_vector = text_stats(open_text(i,folder_name,v ,lower),corpus, top)
            corpy = {**corpus, **text_vector}
            files.append(corpy)
            file_names.append(i + '_' + v)

    vect_ar = np.array(files)        
    return vect_ar,file_names #outputs vector file containing document vector for each text file

def generate_dataframe(vectors, file_names, vocabulary):
    
    '''pandas dataframe generator'''

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    df = pd.DataFrame(data=[i for i in vectors], index=[i for i in file_names], columns = [i for i in vocabulary])
    return df #returns dataframe using data from document vectors, filenames an vocabulary

def find_duplicates(data_frame):

    '''Built in pandas function to remove and list duplicated document vectors'''

    final_matrix = data_frame.drop_duplicates()
    dups_list = data_frame[data_frame.duplicated()].index.tolist()
    
    print("The following indexes have been dropped as duplicates: ")
    for i in dups_list:
        print(i) #printing the duplicated vectors
    
    return final_matrix #final data_frame to be exported 

def generate_tdidf(vector_matrix, filenames, truncated):
    
    '''Function that transforms document vector into tfidf vector using built TfidfTransformer'''

    matrix = []
    print(vector_matrix[0])
    for i in vector_matrix:
        matrix.append(list(i.values()))
    
    #if truncated is false returns matrix to be passed to dataframe function 
    if truncated == False:
        tfidf_matrix = TfidfTransformer().fit_transform(X=matrix).toarray()
        df = pd.DataFrame(data=tfidf_matrix, index=[i for i in filenames])
    else:
        tfidf_matrix = TfidfTransformer().fit_transform(X=matrix).toarray()
        return tfidf_matrix
    return df #dataframe to be sent to output

def parsing(parsing):

    #function that stores all the args definitions

    args = parsing
    
    if args.B:
        n = args.B
        top = True #Topn words
    else:
        n = 10
        top = False
    if args.T: 
        tdidf = True #tfidf
    else:
        tdidf = False

    if args.S:
        truncated = True #truncation
        m = args.S
    else:
        truncated = False
        m = None

    
    return top, n, tdidf, truncated, m , args.filename #returns the bools to be passed to other functions

def truncated_svd(vectors, m,tdidf, vocabulary, filenames):
    
    '''function used to reduce dimensionality of a vector. Uses builtin Truncated SVD
    on document vectors.'''

    data = []
    if tdidf == False:
        for i in vectors:
            data.append(list(i.values()))
        
        svd_data = pd.DataFrame(TruncatedSVD(n_components=m).fit_transform(X=data), index=[i for i in filenames])
    else:
        svd_data = pd.DataFrame(TruncatedSVD(n_components=m).fit_transform(X=vectors), index=[i for i in filenames])
    return svd_data

def output_file(data_frame_raw, vocabulary, file_names, top, tdidf, truncated_svd, m):
   
    '''Function for outputting files, output file name changes depending on what 
    arguments are present'''
    if truncated_svd == False and tdidf == False and top == False:
        #one file containing a term-document matrix with no vocabulary restriction and no other transformations.
        #python3 gendoc.py reuters-topics
        data_frame_raw.to_csv(path_or_buf='vectorfileraw.csv', index = [i for i in file_names], columns= [i for i in vocabulary])

    elif top == True and truncated_svd == False and tdidf == False:
        #one file containing a term-document matrix with a vocabulary restriction of your choice, but allowing significantly fewer terms than the total number and no other transformations.
        #python3 gendoc.py -B20 reuters-topics
        data_frame_raw.to_csv(path_or_buf='vectorfile_top20.csv',index=[i for i in file_names])

    elif tdidf == True and truncated_svd == False and top == False:
        #one file containing a term-document matrix with no vocabulary restriction, but with tf-idf applied.
        #python3 gendoc.py -T reuters-topics
        data_frame_raw.to_csv(path_or_buf='vectorfileraw_idf.csv', index=[i for i in file_names])
    
    elif tdidf == True and top == True and truncated_svd == False:
        #one file containing the same vocabulary restriction as in (2), but with tf-idf applied.
        #python3 gendoc.py -B20 -T reuters-topics
        data_frame_raw.to_csv(path_or_buf='vectorfile_tdidf_top20.csv',index=[i for i in file_names])
    
    elif truncated_svd == True and m == 100 and tdidf == False:
        #one file with no vocabulary restriction, with truncated SVD applied to 100 dimensions.
        #python3 gendoc.py -S100 reuters-topics
        data_frame_raw.to_csv(path_or_buf='vectorfile_truncatedm100.csv',index=[i for i in file_names])
    
    elif truncated_svd == True and m == 1000 and tdidf == False:
        #one file with no vocabulary restriction, with truncated SVD applied to 1000 dimensions.
        #python3 gendoc.py -S1000 reuters-topics
        data_frame_raw.to_csv(path_or_buf='vectorfile_truncatedm1000.csv', index=[i for i in file_names])
    
    elif truncated_svd == True and m == 100 and tdidf == True:
        #one file as in (3), but with truncated SVD applied to 100 dimensions.
        #python3 gendoc.py  -T -S100 reuters-topics
        data_frame_raw.to_csv(path_or_buf='vectorfile_truncatedm100_tdidf.csv', index=[i for i in file_names])
    
    elif truncated_svd == True and m == 1000 and tdidf == True:
        #one file as in (3), but with truncated SVD applied to 1000 dimensions.
        #python3 gendoc.py  -T -S1000 reuters-topics
        data_frame_raw.to_csv(path_or_buf='vectorfiletruncated_m1000_tdidf.csv', index=[i for i in file_names])
    
    else:
        data_frame_raw.to_csv(path_or_buf='vectorfile_unknown.csv', index = [i for i in file_names], columns= [i for i in vocabulary])
         
def main(top, n , tdidf, truncated, m, foldername):
    
    '''Main function for calling all other functions taking into account 
    which arguments have been accepted'''

    print("Generating Corpus")
    corpus,vocabulary = generate_corpus(foldername,top, n)
    print("Corpus Generated\nGenerating Vectors")
    vectors,file_names = generate_files(foldername, corpus,top, n)
    print("Vectors Generated")
    
    if tdidf == True and truncated == False:
        print('Tdidf = True and truncated = False')
        print('generating tdidf vectors')
        tidf_vectors = generate_tdidf(vectors, file_names, truncated)
        print('generating tdidf dataframe')
        print("Removing Duplicates")
        final_data = find_duplicates(tidf_vectors)
        print("outputting file")
        output_file(final_data, vocabulary, file_names, top, tdidf, truncated, m)
        return "done"
    
    if truncated == True and tdidf == True:
        print('Tdidf = True and Truncated = True')
        print('generating tdidf vectors')
        tdidf_vectors = generate_tdidf(vectors, vocabulary, truncated)
        print("Truncating")
        print("Generating Dataframe")
        data_frame = truncated_svd(tdidf_vectors, m, tdidf, vocabulary, file_names)
        print("Removing Duplicates")
        final_data = find_duplicates(data_frame)
        print("outputting file")
        output_file(final_data, vocabulary, file_names, top, tdidf, truncated, m)
        return "done"

    if truncated == True and tdidf == False:
        print('Tdidf = False and Truncated = True')
        print("Truncating")
        print("Generating Dataframe")
        data_frame = truncated_svd(vectors, m, tdidf, vocabulary, file_names)
        print("Removing Duplicates")
        final_data = find_duplicates(data_frame)
        print("outputting file")
        output_file(final_data, vocabulary, file_names, top, tdidf, truncated, m)
        return "done"

    print("Generating Dataframe")
    data_frame = generate_dataframe(vectors,file_names, vocabulary)
    print("Removing Duplicates")
    final_data = find_duplicates(data_frame)
    print("outputting file")
    output_file(final_data, vocabulary, file_names, top, tdidf, truncated, m)
    return "done"
    
if __name__ == "__main__":
    
    '''initialising arguments and running main function'''
    top, n, tdidf, truncated, m , foldername = parsing(parser.parse_args())
    main(top, n , tdidf, truncated, m, foldername)

 