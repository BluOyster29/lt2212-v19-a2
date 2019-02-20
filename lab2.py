import argparse, operator, os, numpy as np
import pandas
 
from nltk import word_tokenize, FreqDist
"""
parser = argparse.ArgumentParser(description='Reads a text and give us analysis')
parser.add_argument("filename", help="takes file name as argument", type=str)
parser.add_argument("--top", help="shows top words and counts", type=int)
parser.add_argument("--to-lower", help="processes the text as lower case", action = "store_true")
parser.add_argument("-T", help="counts will be transformed from raw counts into tf-idf")
parser.add_argument("-S", help="sklearn will do something")
"""
def open_text(subfolder, folder_name, file_path, lower):
    file =folder_name + "/" + subfolder + '/' + file_path
    
    tokenised = []
    with open(file, "r", encoding="utf-8") as text:
        if lower == True:
            for line in text:
                tokenised += word_tokenize(line.lower())
        else:
            for line in text:
                tokenised += word_tokenize(line)
    
    return tokenised

def text_stats(list_tokens, topn):
    corpus_freqs = dict(FreqDist(list_tokens))
    return corpus_freqs
    """sorted_x = sorted(corpus_freqs.items(), key=operator.itemgetter(1), reverse=True)
    for i in sorted_x[:topn]:
        print(i[1], "", str(i[0]))
    """

def generate_files(folder_name):
    lower = True #change this to argparser 
    subfolders = [i for i in os.listdir(folder_name)]
    files = []
    for i in subfolders:
        files.append([text_stats(open_text(i,folder_name, v, lower), 10)for v in os.listdir(folder_name + "/" + i)])
    
    return files
    

def gen_document(vectors):
    pass

#args = parser.parse_args()



if __name__ == "__main__":

    lower = True
    n = 10
    """
    if args.to_lower:
        lower = True
    else:
        lower = False
    if args.top:
        n = args.top
    else:
        pass
        n = 10
    """
    #test_file = open_text(args.filename, lower)
    #text_stats(test_file, n)

    foldername = "reuters-topics" #change this to an argparse 
    vectors = generate_files(foldername)
    print(vectors[0])


    
    
    texticle = open("vectors.txt", "w")
    texticle.write(str(outpu))
    texticle.close()
    