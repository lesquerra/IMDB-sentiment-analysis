import os
import time
import re
import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier 
       
train_path = "./lib/publicdata/aclImdb/train/"
train_file = "imdb_tr.csv"
test_path = "./lib/publicdata/imdb_te.csv"

def remove_stopwords(sentence, stopwords):
    """ Remove Stopwords function
    
    Args:
        sentence: String containing the full sentence to be cleansed
        stopwords: List of words to be removed from the sentence (stopwords and other common words)
        
    Returns: Returns clean sentence string after removing specified words
    
    """
    sentencewords = sentence.split()
    resultwords  = [word for word in sentencewords if word.lower() not in stopwords]
    result = ' '.join(resultwords)
    return result

def remove_special_chars(sentence):
    """ Remove Special Characters function
    
    Args:
        sentence: String containing the full sentence to be cleansed
        
    Returns: Returns clean sentence string after removing special characters
    
    """
    result = re.sub(r"[^a-zA-Z0-9.]+", ' ', re.sub('\.\.+', ' ', sentence))
    return result
    
def imdb_data_preprocess(inpath, outpath = "./", fname = train_file, mix = False):
    """ Data Preprocessing function
    
    Args:
        inpath: Folder path containing the raw data files to be compiled
        outpath: Folder path where the compiled dataset should be saved
        fname: Filename for the compiled dataset
        mix: Boolean indicating whether the original file order should be shuffled when compiling the dataset
        
    Returns: No specific return value. Saves the compiled dataset as a csv file with the specified filename in outpath
    
    """
    stopwords = open("stopwords.en.txt", 'r' , encoding="ISO-8859-1").read()
    stopwords = stopwords.split("\n")
    
    indices, text, rating = [], [], []
    
    i =  0 
    for filename in os.listdir(inpath + "pos"):
        data = open(inpath + "pos/" + filename, 'r' , encoding = "ISO-8859-1").read()
        data = remove_stopwords(data, stopwords)
        data = remove_special_chars(data)
        indices.append(i)
        text.append(data)
        rating.append("1")
        i += 1
        
    for filename in os.listdir(inpath + "neg"):
        data = open(inpath + "neg/" + filename, 'r' , encoding = "ISO-8859-1").read()
        data = remove_stopwords(data, stopwords)
        data = remove_special_chars(data)
        indices.append(i)
        text.append(data)
        rating.append("0")
        i += 1
    
    dataset = list(zip(indices, text, rating))

    if mix:
        np.random.shuffle(dataset)
    
    df = pd.DataFrame(data = dataset, columns=['row_number', 'text', 'polarity'])
    df.to_csv(outpath + fname, index = False, header = True)
    
    pass

def split_data(name, is_train = True):
    """ Split response variable from train data function
    
    Args:
        name: Filename containing the dataset to be split
        is_train: Boolean indicating whether the file is train (containing response variable) or test (train = True; test = False)
        
    Returns: Returns a dataframe containing explanatory variables and, if is_train = True, a vector containing the response variable
    
    """
    data = pd.read_csv(name, header = 0, encoding = 'ISO-8859-1')
    X = data['text']
    if is_train:
        Y = data['polarity']
        return X, Y
    return X

def accuracy(y_train, y_pred):
    """ Accuracy calculation function
    
    Args:
        y_train: Train data vector containing the actual polarity (response variable)
        y_pred: Data vector containing the predicted polarity (response variable)
        
    Returns: Returns the percentual accuracy of the predicted train polarities
    
    """
    assert (len(y_train) == len(y_pred))
    num =  sum([1 for i, word in enumerate(y_train) if y_pred[i] == word])
    n = len(y_train)  
    return (num*100)/n

def unigram_representation(data):
    """ Unigram Data Representation transformation
    
    Args:
        data: Pandas series object containing the reviews to be transformed
        
    Returns: Count vectorizer object containing the unigram representation of the data
    
    """
    vec = CountVectorizer()
    vec = vec.fit(data)
    return vec	

def bigram_representation(data):
    """ Bigram Data Representation transformation
    
    Args:
        data: Pandas series object containing the reviews to be transformed
        
    Returns: Count vectorizer object containing the bigram representation of the data
    
    """
    vec = CountVectorizer(ngram_range=(1,2))
    vec = vec.fit(data)
    return vec

def gradient_descent(x_train, y_train, x_test, n_iter):
    """ Gradient Descent algorithm
    
    Args:
        x_train: Train sparse data matrix containing the reviews' representation
        y_train: Train data vector containing the actual polarity (response variable) of the reviews
        x_test: Test sparse data matrix containing the reviews' representation
        n_iter: Number of iterations
        
    Returns: Data vector containing the predicted polarity (response variable)
    
    """
    clf = SGDClassifier(loss = "hinge", penalty = "l1", max_iter = n_iter)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred

def tfidf(data):
    """ TF-IDF Data transformation
    
    Args:
        data: Sparse data matrix containing the unigram/bigram reviews representation to be transformed
        
    Returns: Tfidf Transformer object containing the TF-IDF representation of the data
    
    """
    tf_trans = TfidfTransformer()
    tf_trans = tf_trans.fit(data)
    return tf_trans

def write_output(fname, data):
    """ Write Output function
    
    Args:
        fname: Filename for the output results to be saved
        data: Array containing the predicted polarity of the test reviews
        
    Returns: No specific return value. Saves the predicted responses as a plain text file with the specified filename
    
    """
    data = '\n'.join(str(i) for i in data)
    file = open(fname, 'w')
    file.write(data)
    file.close()
    pass 

if __name__ == "__main__":

    start = time.time()
    
    print("Preprocessing Files")
    imdb_data_preprocess(inpath = train_path, mix = True)
    [x_train, y_train] = split_data(name = train_file)
    x_test = split_data(name = test_path, is_train = False)
    
    print ("Transforming Unigram Data")
    uni_rep = unigram_representation(x_train)
    x_train_uni = uni_rep.transform(x_train)
    uni_tfidf = tfidf(x_train_uni)
    x_train_tf_uni = uni_tfidf.transform(x_train_uni)
    
    print ("Transforming Bigram Data")    
    bi_rep = bigram_representation(x_train)
    x_train_bi = bi_rep.transform(x_train)
    bi_tfidf = tfidf(x_train_bi)
    x_train_tf_bi = bi_tfidf.transform(x_train_bi)
    
    print ("Fitting Unigram model")
    x_test_uni = uni_rep.transform(x_test)
    y_test_uni = gradient_descent(x_train_uni, y_train, x_test_uni, 50)
    write_output(fname = "unigram.output.txt", data = y_test_uni)
    
    print ("Fitting Bigram model")    
    x_test_bi = bi_rep.transform(x_test)
    y_test_bi = gradient_descent(x_train_bi, y_train, x_test_bi, 20)
    write_output(fname = "bigram.output.txt", data = y_test_bi)
    
    print ("Fitting Unigram Tf-idf")
    x_test_tf_uni = uni_tfidf.transform(x_test_uni)
    y_test_tf_uni = gradient_descent(x_train_tf_uni, y_train, x_test_tf_uni, 20)
    write_output(fname = "unigramtfidf.output.txt", data = y_test_tf_uni)
    
    print ("Fitting Bigram Tf-idf")	
    x_test_tf_bi = bi_tfidf.transform(x_test_bi)
    y_test_tf_bi = gradient_descent(x_train_tf_bi, y_train, x_test_tf_bi, 20)
    write_output(fname = "bigramtfidf.output.txt", data = y_test_tf_bi)
    
    print ("Time taken: ", time.time()-start, " seconds")