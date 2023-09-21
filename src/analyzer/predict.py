import numpy as np
import orjson
from src.analyzer.custom_serielizer import JSdecoded
# from src.analyzer.train_model import predict_tweet
from src.analyzer.utils import process_tweet

def sigmoid(z): 
    '''
    Input:
        z: is the input (can be a scalar or an array)
    Output:
        h: the sigmoid of z
    '''
    h = 1 / (1+(1/np.exp(z)))
    return h


def extract_features(tweet, freqs, process_tweet=process_tweet):
    '''
    Input: 
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output: 
        x: a feature vector of dimension (1,3)
    '''
    # process_tweet tokenizes, stems, and removes stopwords
    word_l = process_tweet(tweet)
    
    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3)) 
    
    #bias term is set to 1
    x[0,0] = 1 
    
    ### START CODE HERE ###
    
    # loop through each word in the list of words
    for word in word_l:
        # increment the word count for the positive label 1
        x[0,1] += freqs.get((word, 1.0)) if freqs.get((word, 1.0)) else 0
        
        # increment the word count for the negative label 0
        x[0,2] += freqs.get((word, 0.0)) if freqs.get((word, 0.0)) else 0
        
    ### END CODE HERE ###
    assert(x.shape == (1, 3))
    return x

def predict_tweet(tweet, freqs, theta):
    '''
    Input: 
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output: 
        y_pred: the probability of a tweet being positive or negative
    '''    
    # extract the features of the tweet and store it into x
    x = extract_features(tweet, freqs)
    # make the prediction using x and theta
    # print('features', x)
    # print('dot', np.dot(x,theta))
    y_pred = sigmoid(np.dot(x,theta))
    # print('y_pred', y_pred)
    return y_pred

with open("model_parameters.json", "rb") as file:
    global freqs
    global theta
    data = file.read()
    deserialized_parameters = orjson.loads(data)
    freqs = JSdecoded(deserialized_parameters.get('freqs'))
    theta = np.array(deserialized_parameters.get('theta'))

def predict_hashtag_sentiment(tweet:str) -> float:
    x =  predict_tweet(tweet, freqs, theta)
    return x[0]