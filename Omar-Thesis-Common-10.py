
# coding: utf-8

# In[1]:


get_ipython().magic('reset -f')

import os

rootPath = "semeval2017-task8-dataset"
TweetsRootPath = rootPath + "/" + "rumoureval-data"
TrainingDataPath = rootPath + "/" + "traindev"
SubTaskATrainFile = "rumoureval-subtaskA-train.json"
SubTaskADevFile = "rumoureval-subtaskA-dev.json"
SubTaskADevResultsFile = "rumoureval-subtaskA-dev-results.json"
SubTaskATestResultsFile = "rumoureval-subtaskA-test-results.json"
SubTaskAGoldResultsFile = "rumoureval-subtaskA-gold-results.json"
SubTaskATrainFilePath = os.path.join(TrainingDataPath,SubTaskATrainFile)
SubTaskADevFilePath = os.path.join(TrainingDataPath,SubTaskADevFile)
SubTaskADevResultsFilePath = os.path.join(TrainingDataPath,SubTaskADevResultsFile)
SubTaskATestResultsFilePath = os.path.join(TrainingDataPath,SubTaskATestResultsFile)
SubTaskAGoldResultsFilePath = os.path.join(TrainingDataPath,SubTaskAGoldResultsFile)

SubTaskBTrainFile = "rumoureval-subtaskB-train.json"
SubTaskBDevFile = "rumoureval-subtaskB-dev.json"
SubTaskBDevResultsFile = "rumoureval-subtaskB-dev-results.json"
SubTaskBTestResultsFile = "rumoureval-subtaskB-test-results.json"
SubTaskBGoldResultsFile = "rumoureval-subtaskB-gold-results.json"
SubTaskBTrainFilePath = os.path.join(TrainingDataPath,SubTaskBTrainFile)
SubTaskBDevFilePath = os.path.join(TrainingDataPath,SubTaskBDevFile)
SubTaskBDevResultsFilePath = os.path.join(TrainingDataPath,SubTaskBDevResultsFile)
SubTaskBTestResultsFilePath = os.path.join(TrainingDataPath,SubTaskBTestResultsFile)
SubTaskBGoldResultsFilePath = os.path.join(TrainingDataPath,SubTaskBGoldResultsFile)

testRootPath = "Test"
TestTweetsRootPath = testRootPath + "/" + "rumoureval2017-test/semeval2017-task8-test-data"

#Gold Results data
goldRootPath = "D:/Master/Thesis/Rumour Detection/Task/Test/Gold"
SubTaskAGoldFile = "subtaskA_gold.json"
SubTaskBGoldFile = "subtaskB_gold.json"
SubTaskAGoldFilePath = goldRootPath + "/" + SubTaskAGoldFile
SubTaskBGoldFilePath = goldRootPath + "/" + SubTaskBGoldFile


# In[6]:


import json
from pprint import pprint

with open(SubTaskATrainFilePath) as data_file:    
    TaskATrainData = json.load(data_file)

with open(SubTaskADevFilePath) as data_file:    
    TaskADevData = json.load(data_file)
    
with open(SubTaskBTrainFilePath) as data_file:    
    TaskBTrainData = json.load(data_file)

with open(SubTaskBDevFilePath) as data_file:    
    TaskBDevData = json.load(data_file)
    
with open(SubTaskAGoldFilePath) as data_file:    
    TaskAGoldData = json.load(data_file)
    
with open(SubTaskBGoldFilePath) as data_file:    
    TaskBGoldData = json.load(data_file)
#pprint(data)
#TaskATrainData["544284128615473152"]


# In[3]:


def GetTweets(rootPath,TweetsFolderName):
    TweetsJSON = { }
    #print("rootPath = " + rootPath)
    for root, subFolders, files in os.walk(rootPath):
        #print("root = " + root)
        for folder in subFolders:
            if folder == TweetsFolderName:
                srcTweetDir = os.path.join(root,folder)
                #print("srcTweetDir = " + srcTweetDir)
                for  TweetRootPath, _,TweetFileNames in os.walk(srcTweetDir):
                    for TweetFileName in TweetFileNames:
                        filePath = os.path.join(TweetRootPath,TweetFileName)
                        #print("filePath = " + filePath)
                        with open(filePath) as data_file:
                            data = json.load(data_file)
                            tweetID = TweetFileName[0:18]
                            TweetsJSON[tweetID] = data
    return TweetsJSON


# In[4]:


SrcTweetsMap = GetTweets(TweetsRootPath,"source-tweet")
ReplyTweetsMap = GetTweets(TweetsRootPath,"replies")
#print(SrcTweetsMap["553164985460068352"])

#print(ReplyTweetsMap["581219554912800768"])


# In[5]:


SrcTestTweetsMap =  GetTweets(TestTweetsRootPath,"source-tweet")
ReplyTestTweetsMap = GetTweets(TestTweetsRootPath,"replies")


# In[ ]:


def GetTweetObj(ID,IsTest):
    tweetObj = None
    
    if IsTest:
        if ID in ReplyTestTweetsMap:
            tweetObj = ReplyTestTweetsMap[ID].copy()
            tweetObj[ATTR_IS_SRC] = 1;
            #FoundReply = FoundReply + 1
        elif ID in SrcTestTweetsMap:
            tweetObj = SrcTestTweetsMap[ID].copy()
            tweetObj[ATTR_IS_SRC] = 0;
    else:
        if ID in ReplyTweetsMap:
            tweetObj = ReplyTweetsMap[ID].copy()
            tweetObj[ATTR_IS_SRC] = 1;
            #FoundReply = FoundReply + 1
        elif ID in SrcTweetsMap:
            tweetObj = SrcTweetsMap[ID].copy()
            tweetObj[ATTR_IS_SRC] = 0;
            #FoundSrc = FoundSrc + 1
        #else:
            #NotFound = NotFound + 1
            #print("tweet not found: " + key)
            
    return tweetObj


# In[6]:


#Tweets structure
import ntpath
def GetStructs(rootPath,StructFileName):
    StructJSON = { }
    for root, subFolders, files in os.walk(rootPath):
        for file in files:
            if file == StructFileName:
                tweetID = ntpath.basename(root)
                StructFilePath = os.path.join(root,file)
                with open(StructFilePath) as data_file:
                    data = json.load(data_file)
                    StructJSON[tweetID] = data
    return StructJSON

StructsJSON = GetStructs(TweetsRootPath,"structure.json")
TestStructsJSON = GetStructs(TestTweetsRootPath,"structure.json")


# In[7]:


#Get Dict of source tweets for each tweet
#print(type(StructsJSON["529653029747064832"]))
#print(StructsJSON["529653029747064832"])

#Dict of tweet and the value is the source tweet
def ExtractSrcTweetsRecursive(SrcTweetKey,CurrentDict):
    ResultDict = {}
    if type(CurrentDict) is dict:
        for key,val in CurrentDict.items(): 
            ResultDict[key] = SrcTweetKey
            ResultDict.update(ExtractSrcTweetsRecursive(SrcTweetKey,val))
    return ResultDict

SrcTweetsDict = {}
for key,val in StructsJSON.items():
    SrcTweetsDict.update(ExtractSrcTweetsRecursive(key,val))
    SrcTweetsDict[key] = key

SrcTestTweetsDict = {}
for key,val in TestStructsJSON.items():
    SrcTestTweetsDict.update(ExtractSrcTweetsRecursive(key,val))
    SrcTestTweetsDict[key] = key

#print(len(SrcTweetsDict))
#print(len(SrcTestTweetsDict))


# In[9]:


#Getting python files used
get_ipython().magic('run QuestionDetector.py')

#Preparing utility objects
questionDetector = QuestionDetector()

from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()

import string
from nltk.corpus import stopwords

punctuation = list(string.punctuation)
StopWords = stopwords.words('english') + punctuation + ['rt', 'via']
StopWords.sort()

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import VectorizerMixin

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2


from sklearn import metrics
from nltk.corpus import opinion_lexicon

from sklearn import cross_validation
#from sklearn import  model_selection

from sklearn.metrics import fbeta_score,make_scorer,accuracy_score


import re

from sklearn.decomposition import TruncatedSVD

import time
from time import gmtime, strftime,strptime
from datetime import datetime
TWITTER_DATE_FORMAT = '%a %b %d %H:%M:%S +0000 %Y'
MY_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

from sklearn import preprocessing

get_ipython().magic('matplotlib notebook')

import matplotlib.pyplot as plt

import math
from collections import Counter
#Cosine similarity to be used
def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


# In[16]:


#Stemmer

def build_tokenizer():
    """Return a function that splits a string into a sequence of tokens"""
    token_pattern=r"(?u)\b\w\w+\b"
    token_pattern = re.compile(token_pattern)
    return lambda doc: token_pattern.findall(doc)


def _word_ngrams(tokens, stop_words=None):
    ngram_range = (1,2)
    """Turn tokens into a sequence of n-grams after stop words filtering"""
    # handle stop words
    if stop_words is not None:
        tokens = [w for w in tokens if w not in stop_words]

    # handle token n-grams
    min_n, max_n = ngram_range
    if max_n != 1:
        original_tokens = tokens
        tokens = []
        n_original_tokens = len(original_tokens)
        for n in range(min_n,
                        min(max_n + 1, n_original_tokens + 1)):
            for i in range(n_original_tokens - n + 1):
                tokens.append(" ".join(original_tokens[i: i + n]))

    return tokens
    
#%run MyStemmer.py
from nltk.stem.porter import PorterStemmer
#from nltk.stem import SnowballStemmer

#stemmer = MyStemmer()
stemmer = PorterStemmer()
#stemmer = nltk.stem.SnowballStemmer('english')

def simpleStem(word):
    
    l = len(word)
    if l > 3:
        if word.endswith('es'):
            print('before word = ' + word)
            word = word[0:l-2]
            print('after word = ' + word)
        elif word.endswith('s'):
            print('before word = ' + word)
            word = word[0:l-1]
            print('after word = ' + word)
    return word
        

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        #stem= item
        stem = stemmer.stem(item)
        #stem = simpleStem(item)
        
        stemmed.append(stem)
    return stemmed

def tokenize_stem(text):
    tokenize = build_tokenizer()
    
    tokens = _word_ngrams(tokenize(text), StopWords)
    #tokens = nltk.word_tokenize(text)
    #tokens = tknzr.tokenize(text)
    stems = stem_tokens(tokens, stemmer) 
    return stems


# In[ ]:


#Extract pos & neg words
PosLexicon = opinion_lexicon.positive()
NegLexicon = opinion_lexicon.negative()

type(PosLexicon)

PosLexiconList = []
NegLexiconList = []
for word in PosLexicon:
    PosLexiconList.append(word)
for word in NegLexicon:
    NegLexiconList.append(word)
#print(len(PosLexicon))
#print(len(NegLexiconList))


# In[ ]:


#Sentiment Analyzer using movie reviews corprus
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews

class SentimentAnalyzer():

    def word_feats(self,words):
        return dict([(word, True) for word in words])
    
    def __init__(self):
    
        negids = movie_reviews.fileids('neg')
        posids = movie_reviews.fileids('pos')

        negfeats = [(self.word_feats(movie_reviews.words(fileids=[f])), 0) for f in negids]
        posfeats = [(self.word_feats(movie_reviews.words(fileids=[f])), 1) for f in posids]

        trainfeats = negfeats + posfeats

        self.classifier = NaiveBayesClassifier.train(trainfeats)
#         print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
#         classifier.show_most_informative_features()
        
    def classify(self,sent):
        
        feat = self.word_feats(tknzr.tokenize(sent))
        
        return self.classifier.classify(feat)
        
SentAnalyzer = SentimentAnalyzer()


# In[ ]:


#Fooling around Using NLTK

# import nltk
# #nltk.download('punkt') #Used for tokenize
# #nltk.download('stopwords')
# #nltk.download()

# from nltk.tokenize import TweetTokenizer
# tknzr = TweetTokenizer()

# tweet = 'RT @marcobonzanini: just an example! :D http://example.com #NLP'

# tokens = tknzr.tokenize(tweet)
# print(tokens)

# from nltk.corpus import stopwords
# import string
 
# punctuation = list(string.punctuation)
# stop = stopwords.words('english') + punctuation + ['rt', 'via']
# terms_stop = [term for term in tokens if term not in stop]
# print(terms_stop)

# from nltk import bigrams
# terms_bigram = list(bigrams(terms_stop))
# print(terms_bigram)

