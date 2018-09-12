
# coding: utf-8

# In[1]:


#%run Omar-Thesis-Common-7.ipynb

get_ipython().magic('run "Omar Thesis-7A.ipynb"')


# In[36]:


#Extracting tweets and and their deep features
start_time = time.time()

target_names = ["false","true","unverified"]
le.fit(target_names)

TrainData,DevData = {} , {}

#Deny_Words = ['not true', 'shut', 'not', "don't agree","impossible",'false','lies']
Deny_Words = ['not true', 'shut', 'not', "don't agree","impossible",'false']
Deny_Words_Set = set(Deny_Words)
Support_Words = ['true', 'exactly','yes','indeed','agree','omg','know']
Support_Words_Set = set(Support_Words)
URL_Words = ['http://','https://']
URL_Words_Set = set(URL_Words)

ID_FALSE = 0
ID_TRUE = 1
ID_UNVERIFIED = 2

DENY_WORD_FEATURE = "deny_word"
SUPPORT_WORD_FEATURE = "support_word"
QUEST_WORD_FEATURE = "quest_word"
HASHTAG_EXISTS_FEATURE = "hashtag_exists"
URL_EXISTS_FEATURE = "url_exists"
SENTIMENT_WORD_FEATURE = "sentiment_word"
POS_WORDS_COUNT = "pos_words_count"
NEG_WORDS_COUNT = "neg_words_count"
NEUT_WORDS_COUNT = "neut_words_count"
FT_USER_VERIFIED = "user_verified"
FT_USER_FOLLOWERS_COUNT = "user_followers"
FT_USER_TWEETS_COUNT = "user_tweets"
FT_USER_FRIENDS_COUNT = "user_friends"
FT_PHOTO_EXISTS = "photo_exists"
FT_DAYS_SINCE_USER_CREATION = "days_since_user_creation"
FT_PERC_OF_QUERIES = "perc_of_queries"
FT_PERC_OF_DENIES = "perc_of_denies"
FT_PERC_OF_SUPPORT = "perc_of_support"

ATTR_ID = "id"
ATTR_IS_SRC = "is_src"
ATTR_CATEGORY = "category"
ATTR_TEXT = "text"
ATTR_REPLY_STATUS_ID = "in_reply_to_status_id"
ATTR_USER = "user"
ATTR_USER_VERIFIED = "verified"
ATTR_NEW_TEXT = "new_text"
ATTR_FOLLOWERS_COUNT = "followers_count"
ATTR_TWEETS_COUNT = "statuses_count"
ATTR_RETWEET_COUNT = "retweet_count"
ATTR_FRIENDS_COUNT = "friends_count"
ATTR_EXT_ENTITIES = "extended_entities"
ATTR_MEDIA = "media"
ATTR_MEDIA_TYPE = "type"
ATTR_MEDIA_TYPE_PHOTO = "photo"
ATTR_USER_CREATED_AT = "created_at"

currentTime = time.strftime(MY_DATE_FORMAT, gmtime())
currentTime = datetime.strptime(currentTime, MY_DATE_FORMAT)

def ExtractTweetFeatures(tweet):
    posWordsCount = 0
    negWordsCount = 0
        
    tweet[ATTR_NEW_TEXT] =  tweet[ATTR_TEXT]

    #Remove URLs:
    #URLs show something in common between tweets, so removing them produced a worse result.
    #tweet[ATTR_NEW_TEXT] = re.sub(r"http\S+", "", tweet[ATTR_TEXT])

    tweetText = tweet[ATTR_NEW_TEXT].lower()
    #tweetText = tweet[ATTR_NEW_TEXT]

    tokens = tknzr.tokenize(tweetText)
    tokensSet = set(tokens)

    #Extract features
    if questionDetector.IsQuestion( tweet[ATTR_NEW_TEXT]) == True:
        tweet[QUEST_WORD_FEATURE] = 1
    else:
        tweet[QUEST_WORD_FEATURE] = 0

    #if any(word in tweetText for word in Deny_Words):
    if (Deny_Words_Set.intersection(tokensSet)):
        tweet[DENY_WORD_FEATURE] = 1
        #tweet[DENY_WORD_FEATURE] = len(Deny_Words_Set.intersection(tokensSet))
        tweet[SUPPORT_WORD_FEATURE] = 0
    elif (Support_Words_Set.intersection(tokensSet)):
        tweet[SUPPORT_WORD_FEATURE] = 1
        #tweet[SUPPORT_WORD_FEATURE] = len(Support_Words_Set.intersection(tokensSet))
        tweet[DENY_WORD_FEATURE] = 0
    else:
        tweet[DENY_WORD_FEATURE] = 0
        tweet[SUPPORT_WORD_FEATURE] = 0

    #Hashtag
    if (tweetText.find("#")):
        tweet[HASHTAG_EXISTS_FEATURE] = 1
    else:
        tweet[HASHTAG_EXISTS_FEATURE] = 0

    tweet[URL_EXISTS_FEATURE] = 0
    for word in  tokensSet:
        for urlToken in URL_Words_Set:
            if urlToken in word:
                tweet[URL_EXISTS_FEATURE] = 1
                break

    for word in  tokensSet:            
        if word in PosLexiconList:
            posWordsCount += 1
        elif word in NegLexiconList:
            negWordsCount += 1

    tweet[POS_WORDS_COUNT] = posWordsCount
    tweet[NEG_WORDS_COUNT] = negWordsCount
    tweet[NEUT_WORDS_COUNT] = len(tokensSet)-negWordsCount-posWordsCount

    if posWordsCount > negWordsCount:
        tweet[SENTIMENT_WORD_FEATURE] = 1
    elif negWordsCount > posWordsCount:
        tweet[SENTIMENT_WORD_FEATURE] = 0
    else:
        tweet[SENTIMENT_WORD_FEATURE] = 0.5

    if tweet[ATTR_USER][ATTR_USER_VERIFIED] == True:
        tweet[FT_USER_VERIFIED] = 1
    else:
        tweet[FT_USER_VERIFIED] = 0           

    tweet[FT_USER_FOLLOWERS_COUNT] = tweet[ATTR_USER][ATTR_FOLLOWERS_COUNT]
    tweet[FT_USER_TWEETS_COUNT] = tweet[ATTR_USER][ATTR_TWEETS_COUNT]
    tweet[FT_USER_FRIENDS_COUNT] = tweet[ATTR_USER][ATTR_FRIENDS_COUNT]


    #Compute number of days since user creation feature
    tweetTime = time.strftime(MY_DATE_FORMAT, time.strptime(tweet[ATTR_USER][ATTR_USER_CREATED_AT],TWITTER_DATE_FORMAT))
    tweetTime = datetime.strptime(tweetTime, MY_DATE_FORMAT)
    delta = currentTime - tweetTime
    tweet[FT_DAYS_SINCE_USER_CREATION] = delta.days

    #Check if photo exists
    tweet[FT_PHOTO_EXISTS] = 0
    if ATTR_EXT_ENTITIES in tweet:
        ExtEntities = tweet[ATTR_EXT_ENTITIES]
        if ExtEntities is not None:
            Media = ExtEntities[ATTR_MEDIA]
            if Media is not None:
                for m in Media:
                    if m[ATTR_MEDIA_TYPE] == ATTR_MEDIA_TYPE_PHOTO:
                        tweet[FT_PHOTO_EXISTS] = 1
                        break        

    TweetReplies = TweetRepliesMap[tweet[ATTR_ID]]
    
    queries_count = 0
    support_count = 0
    deny_count = 0
    total_replies_count = 0
    for reply in TweetReplies:
        total_replies_count += 1
        if reply[ATTR_CATEGORY] == ID_QUERY:
            queries_count += 1
        elif reply[ATTR_CATEGORY] == ID_SUPPORT:
            support_count += 1
        elif reply[ATTR_CATEGORY] == ID_DENY:
            deny_count += 1
    
    if total_replies_count == 0:
        tweet[FT_PERC_OF_QUERIES] = 0
        tweet[FT_PERC_OF_DENIES] = 0
        tweet[FT_PERC_OF_SUPPORT] = 0
    else:
        tweet[FT_PERC_OF_QUERIES] = queries_count/total_replies_count
        tweet[FT_PERC_OF_DENIES] = deny_count/total_replies_count
        tweet[FT_PERC_OF_SUPPORT] = support_count/total_replies_count
        
    #print('[%s]: %s %s(%f) %s(%f) %s(%f)' % 
    #     (tweet[ATTR_TEXT],total_replies_count,
    #       queries_count,tweet[FT_PERC_OF_QUERIES],deny_count,tweet[FT_PERC_OF_DENIES],support_count,tweet[FT_PERC_OF_SUPPORT]))
    
    return tweet


def GetTweetsInfo (TaskData,IsTest):
    ResultTweetsList = []
    
    for key, value in TaskData.items():       
        tweet = GetTweetObj(key,IsTest)
        if tweet is None:
            continue
            
        tweet[ATTR_CATEGORY] = le.transform(value)
    
        tweet = ExtractTweetFeatures(tweet)
        
        ResultTweetsList.append(tweet)

    return ResultTweetsList

def ExtractTestData (SrcTestTweetsMap):
    ResultTweetsList = []
    
    for key, value in SrcTestTweetsMap.items():    
        tweet = value
        
        tweet = ExtractTweetFeatures(tweet)
        
        ResultTweetsList.append(tweet)

    return ResultTweetsList

TrainData = GetTweetsInfo(TaskBTrainData,False)
DevData = GetTweetsInfo(TaskBDevData,False)
GoldData = GetTweetsInfo(TaskBGoldData,True)
#print("TEST DATA")
TestData = ExtractTestData(SrcTestTweetsMap)

print("--- %s seconds ---" % (time.time() - start_time))

#TrainData


# In[37]:


#Extracting data into lists for ease of access
TrainTweetsText = []
TrainTweetsCategories = []
TrainTweetsIDs = []
for item in TrainData:
    TrainTweetsIDs.append(item[ATTR_ID])
    TrainTweetsText.append(item[ATTR_NEW_TEXT])
    TrainTweetsCategories.append(item[ATTR_CATEGORY])
    
DevTweetsText = []
DevTweetsCategories = []
DevTweetsIDs = []
for item in DevData:
    DevTweetsIDs.append(item[ATTR_ID])
    DevTweetsText.append(item[ATTR_NEW_TEXT])
    DevTweetsCategories.append(item[ATTR_CATEGORY])

GoldTweetsText = []
GoldTweetsCategories = []
GoldTweetsIDs = []
for item in GoldData:
    GoldTweetsIDs.append(item[ATTR_ID])
    GoldTweetsText.append(item[ATTR_NEW_TEXT])
    GoldTweetsCategories.append(item[ATTR_CATEGORY])
    
TestTweetsText = []
TestTweetsIDs = []
for item in TestData:
    TestTweetsIDs.append(item[ATTR_ID])
    TestTweetsText.append(item[ATTR_NEW_TEXT])


# In[38]:


#Transform the text into word vector
from nltk import word_tokenize

punctuation = list(string.punctuation)
StopWords = stopwords.words('english') + punctuation + ['rt', 'via']
StopWords.sort()

USE_PoS = False
USE_TFIDF = False

if not USE_PoS: 
    if USE_TFIDF:
        print("Using TfidfVectorizer")
        tfidf_vect = TfidfVectorizer(ngram_range=(1,1),stop_words =StopWords,lowercase = False)
        X_train_counts = tfidf_vect.fit_transform(TrainTweetsText)
        X_dev_counts = tfidf_vect.transform(DevTweetsText)
        X_test_counts = tfidf_vect.transform(TestTweetsText)
        X_gold_counts = tfidf_vect.transform(GoldTweetsText)
    else:
        print("Using CountVectorizer")
        #Using Count vectorizer with out PoS
        #count_vect = CountVectorizer(ngram_range=(1,1))
        #count_vect = CountVectorizer(ngram_range=(1,1),stop_words ="english")
        #Stemming causes worse performance
        #count_vect = CountVectorizer(ngram_range=(1,1),stop_words =StopWords,tokenizer=tokenize_stem,lowercase = False)
        count_vect = CountVectorizer(ngram_range=(1,1),stop_words =StopWords,lowercase = False)
        X_train_counts = count_vect.fit_transform(TrainTweetsText)
        #Extract the basic words features for dev data
        X_dev_counts = count_vect.transform(DevTweetsText)
        
        X_test_counts = count_vect.transform(TestTweetsText)
        X_gold_counts = count_vect.transform(GoldTweetsText)
else:
    #Using Dict Vectorizer with PoS
    TrainTweetsTextPoS = []
    for tweetText in TrainTweetsText:
        tokensList = [i for i in word_tokenize(tweetText.lower()) if i not in StopWords]
        PoSTokens = nltk.pos_tag(tokensList) #PoSTokens is a tuple
        TrainTweetsTextPoS.append(dict((y, x) for x, y in PoSTokens))
    
    X_train_counts = dict_vect.fit_transform(TrainTweetsTextPoS)
    
    DevTweetsTextPoS = []
    for tweetText in DevTweetsText:
        tokensList = [i for i in word_tokenize(tweetText.lower()) if i not in StopWords]
        PoSTokens = nltk.pos_tag(tokensList)
        DevTweetsTextPoS.append(dict((y, x) for x, y in PoSTokens))
        
    X_dev_counts = dict_vect.transform(DevTweetsTextPoS)
    
    TestTweetsTextPoS = []
    for tweetText in TestTweetsText:
        tokensList = [i for i in word_tokenize(tweetText.lower()) if i not in StopWords]
        PoSTokens = nltk.pos_tag(tokensList)
        TestTweetsTextPoS.append(dict((y, x) for x, y in PoSTokens))
        
    X_test_counts = dict_vect.transform(TestTweetsTextPoS)
    
    GoldTweetsTextPoS = []
    for tweetText in GoldTweetsText:
        tokensList = [i for i in word_tokenize(tweetText.lower()) if i not in StopWords]
        PoSTokens = nltk.pos_tag(tokensList)
        GoldTweetsTextPoS.append(dict((y, x) for x, y in PoSTokens))
        
    X_gold_counts = dict_vect.transform(GoldTweetsTextPoS)
    
trainFts = sp.csr_matrix(np.zeros((len(TrainData),0)))
devFts = sp.csr_matrix(np.zeros((len(DevData),0)))
testFts = sp.csr_matrix(np.zeros((len(TestData),0)))
goldFts = sp.csr_matrix(np.zeros((len(GoldData),0)))

trainFts = X_train_counts
devFts = X_dev_counts
testFts = X_test_counts
goldFts = X_gold_counts

#Print stats
print (len(TrainTweetsText))
print (type(trainFts))
print (trainFts.shape)
print (devFts.shape)
print (goldFts.shape)

#print(type(trainFts))
#print(trainFts.toarray())

# count = 0
# for f in trainFts.toarray():
#     if f == 1:
#         count = count + 1

# allCount = 0
# for item in TrainData:
#     if item["category"] == ID_QUERY:
#         allCount = allCount + 1
# print("All Queries: " + str(allCount))
# print("Considered questions: " + str(count))


# In[39]:


#Fooling around: Print word features

#if not USE_PoS: 
    #print(count_vect.get_feature_names())
#else:
    #print(dict_vect.get_feature_names())



# In[40]:


#Feature selection
CHI_FS_ON = 0
VT_FS_ON = 0
DR_SVD_ON = 0

plt.close()

if CHI_FS_ON:
    print ('Chi-Squared Feature Selection')
    FeaturesCount = 9687
    #FeaturesCount = 9650

    ch2 = SelectKBest(chi2, k=FeaturesCount)

    print (trainFts.shape)

    trainFts = ch2.fit_transform(trainFts, TrainTweetsCategories)
    devFts = ch2.transform(devFts)
    goldFts = ch2.transform(goldFts)

    print (trainFts.shape)


from sklearn.feature_selection import VarianceThreshold

if VT_FS_ON:
    print ('Variance Threshold Feature Selection')
    
    #vtFT = VarianceThreshold(threshold=(0.00023725))
    #vtFT = VarianceThreshold(threshold=(0.2))
    vtFT = VarianceThreshold(0.00025)
    
    print (trainFts.shape)
    
    trainFts = vtFT.fit_transform(trainFts)
    
    print(vtFT.variances_)
    print(min(vtFT.variances_))
    print(max(vtFT.variances_))
    
    _,ax = plt.subplots()
    bins = np.linspace(0.00030,0.008, 1000)
    ax.hist(vtFT.variances_,bins)
        
    devFts = vtFT.transform(devFts)
    
    goldFts = vtFT.transform(goldFts)
    
    print (trainFts.shape)
    
if DR_SVD_ON:
    svd = TruncatedSVD(n_components=200, n_iter=7, random_state=42)
    svd.fit(trainFts)
    #print(svd.explained_variance_ratio_) 
    
    print (trainFts.shape)
    #print(type(trainFts))
    trainFts = sp.csr_matrix(svd.transform(trainFts))
    devFts = sp.csr_matrix(svd.transform(devFts))
    goldFts = sp.csr_matrix(svd.transform(goldFts))
    
    #print(type(trainFts))
    print (trainFts.shape)


# In[41]:


#Add additional features
def AddBinaryFeatureToVector (featureName,features,data,scale=False):
    NewFeatureArr = np.zeros((len(data),1))
    i =0
    for item in data:
        NewFeatureArr[i] = item[featureName]
        i = i + 1
    
    if scale:
        NewFeatureArr = preprocessing.scale(NewFeatureArr)
    
    features = sp.hstack((features, NewFeatureArr), format='csr')
    
    return features

def AddFeature(trainFts,devFts,testFts,goldFts,attrName,scale=False):
    trainFts = AddBinaryFeatureToVector(attrName,trainFts,TrainData,scale)
    devFts = AddBinaryFeatureToVector(attrName,devFts,DevData,scale)
    testFts = AddBinaryFeatureToVector(attrName,testFts,TestData,scale)
    goldFts = AddBinaryFeatureToVector(attrName,goldFts,GoldData,scale)
    return (trainFts,devFts,testFts,goldFts)

#trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,QUEST_WORD_FEATURE)

#trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,DENY_WORD_FEATURE)

#trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,SUPPORT_WORD_FEATURE)

trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,HASHTAG_EXISTS_FEATURE)

trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,URL_EXISTS_FEATURE)

#trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,SENTIMENT_WORD_FEATURE)

#Useful with SGD Classifier
#trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,FT_USER_VERIFIED)

# #User's Followers count
#trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,FT_USER_FOLLOWERS_COUNT,True)

# #User's tweets count
#trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,FT_USER_TWEETS_COUNT,True)

# #User's friends count
#trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,FT_USER_FRIENDS_COUNT,True)

#Photo existance
#trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,FT_PHOTO_EXISTS,True)

#Days since user creation
#trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,FT_DAYS_SINCE_USER_CREATION,True)

trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,FT_PERC_OF_QUERIES)
trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,FT_PERC_OF_DENIES)
trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,FT_PERC_OF_SUPPORT)

print (len(TrainTweetsText))
print (trainFts.shape)
print (devFts.shape)
print (testFts.shape)
print (goldFts.shape)
#print(StopWords)


# In[42]:


#Choose classifier

from sklearn.calibration import CalibratedClassifierCV


#Better: 0.44
#from sklearn.linear_model import SGDClassifier
#clf = SGDClassifier()

#Better: 0.36
from sklearn.svm import LinearSVC
clf = LinearSVC()
#clf = CalibratedClassifierCV(clf) 
#clf = LinearSVC(class_weight="balanced")

#from sklearn.naive_bayes import MultinomialNB
#clf = MultinomialNB()

#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier()

#from sklearn.linear_model import LogisticRegression
#clf = LogisticRegression()

#from sklearn.svm import SVC
#clf = SVC()


# In[43]:


#Cross Validation using train data

resultsArr = np.array(list(TrainTweetsCategories), dtype=int)

scorer = make_scorer(fbeta_score,greater_is_better=True,labels=[ID_FALSE,ID_TRUE,ID_UNVERIFIED],average="macro",beta=2)

scores = cross_validation.cross_val_score(clf, trainFts, resultsArr, cv=5,scoring=scorer)
print(scores)
print("Macro Averaged F1 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scorer = make_scorer(fbeta_score,greater_is_better=True,labels=[ID_FALSE,ID_TRUE,ID_UNVERIFIED],average="weighted",beta=2)

scores = cross_validation.cross_val_score(clf, trainFts, resultsArr, cv=5,scoring=scorer)
print(scores)
print("Weighted Averaged F1 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_validation.cross_val_score(clf, trainFts, resultsArr, cv=5)
print(scores)
print("Normal Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[44]:


#Train Classifier
import numpy as np
array = np.array(list(TrainTweetsCategories), dtype=int)

clf.fit(trainFts,array)


# In[45]:


#Test Classifier
predicted = clf.predict(devFts)
#for doc, category in zip(DevData.keys(), predicted):
#    print('%r => %s' % (doc, category))



# In[46]:


#Evaluate Classifier using dev data

resultsArr = np.array(list(DevTweetsCategories), dtype=int)
print ("Accuracy: " + str(np.mean(predicted == resultsArr)))

print(metrics.classification_report(resultsArr, predicted,target_names=target_names))

print("*Macro F1 Average*: " + str(metrics.f1_score(resultsArr, predicted,average='macro')))
print("Micro F1 Average: " + str(metrics.f1_score(resultsArr, predicted,average='micro')))
print("Weighted F1 Average: " + str(metrics.f1_score(resultsArr, predicted,average='weighted')))


# In[47]:


#Write dev results to file

decisionFunc = clf.decision_function(devFts)

ResultDict = {}
i = 0
while i < len(DevTweetsIDs):
    Conf = max(decisionFunc[i])
    #print(Conf)
    if Conf < -0.2:
        Conf = 0.2
    if Conf < 0:
        Conf = 0.4
    elif Conf < 0.2:
        Conf = 0.6
    else:
        Conf = 0.8
    ResultDict[DevTweetsIDs[i]] = [target_names[predicted[i]], Conf]
    i+= 1
    
with open(SubTaskBDevResultsFilePath, 'w') as outfile:
    json.dump(ResultDict, outfile)

ipy = get_ipython()
ipy.magic("run ScorerB.py \"" + SubTaskBDevFilePath + "\" \"" + SubTaskBDevResultsFilePath + "\"")


# In[48]:


clf = LinearSVC()

# print(type(devFts))
TrainAndDevCategories = TrainTweetsCategories+DevTweetsCategories
TrainAndDevFts = sp.vstack((trainFts,devFts),format='csr')
print(trainFts.shape)
print(devFts.shape)
print(TrainAndDevFts.shape)

CategoriesArr = np.array(list(TrainAndDevCategories), dtype=int)

clf.fit(TrainAndDevFts,CategoriesArr)


# In[49]:


#Write test results to file

print(testFts.shape)
print(len(TestTweetsIDs))

predicted = clf.predict(testFts)

decisionFunc = clf.decision_function(testFts)

ResultDict = {}
i = 0
while i < len(TestTweetsIDs):
    Conf = max(decisionFunc[i])
    #print(Conf)
    if Conf < -0.2:
        Conf = 0.2
    if Conf < 0:
        Conf = 0.4
    elif Conf < 0.2:
        Conf = 0.6
    else:
        Conf = 0.8
    ResultDict[TestTweetsIDs[i]] = [target_names[predicted[i]], Conf]
    i+= 1
    
with open(SubTaskBTestResultsFilePath, 'w') as outfile:
    json.dump(ResultDict, outfile)


# In[50]:


#Write gold results to file

decisionFunc = clf.decision_function(goldFts)

ResultDict = {}
i = 0
while i < len(GoldTweetsIDs):
    Conf = max(decisionFunc[i])
    #print(Conf)
    if Conf < -0.2:
        Conf = 0.2
    if Conf < 0:
        Conf = 0.4
    elif Conf < 0.2:
        Conf = 0.6
    else:
        Conf = 0.8
    ResultDict[GoldTweetsIDs[i]] = [target_names[predicted[i]], Conf]
    i+= 1
    
with open(SubTaskBGoldResultsFilePath, 'w') as outfile:
    json.dump(ResultDict, outfile)

ipy = get_ipython()
ipy.magic("run ScorerB.py \"" + SubTaskBGoldFilePath + "\" \"" + SubTaskBGoldResultsFilePath + "\"")

