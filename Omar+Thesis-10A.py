
# coding: utf-8

# In[1]:


get_ipython().magic('run Omar-Thesis-Common-10.ipynb')


# In[2]:


import urllib

#Extracting tweets and and their deep features
start_time = time.time()

target_names = ["comment","deny", "query","support"]
le.fit(target_names)

TrainData,DevData,TestData,GoldData = [] , [], [], []

#Deny_Words = ['not true', 'shut', 'not', "don't agree","impossible",'false','lies']
Deny_Words = ['not true', 'shut', 'not', "don't agree","impossible",'false']
Deny_Words_Set = set(Deny_Words)
Support_Words = ['true', 'exactly','yes','indeed','agree','omg','know']
Support_Words_Set = set(Support_Words)
URL_Words = ['http://','https://']
URL_Words_Set = set(URL_Words)

ID_COMMENT = 0
ID_DENY = 1
ID_QUERY = 2
ID_SUPPORT = 3

DENY_WORD_FEATURE = "deny_word"
SUPPORT_WORD_FEATURE = "support_word"
QUEST_WORD_FEATURE = "quest_word"
HASHTAG_EXISTS_FEATURE = "hashtag_exists"
URL_EXISTS_FEATURE = "url_exists"
SENTIMENT_WORD_FEATURE = "sentiment_word"
POS_WORDS_COUNT = "pos_words_count"
NEG_WORDS_COUNT = "neg_words_count"
NEUT_WORDS_COUNT = "neut_words_count"
IS_REPLY_FEATURE = "is_reply_tweet"
FT_USER_VERIFIED = "user_verified"
FT_SRC_USER_VERIFIED = "src_user_verified"
FT_DIRECT_SRC_USER_VERIFIED = "direct_src_user_verified"
FT_USER_FOLLOWERS_COUNT = "user_followers"
FT_USER_TWEETS_COUNT = "user_tweets"
FT_COSINE_SIM_SRC_TWEET = "cos_sim_src_tweet"
FT_COSINE_SIM_SRC_DIRECT_TWEET = "cos_sim_src_direct_tweet"
FT_RETWEET_RATIO = "ft_retweet_ratio"
FT_USER_FRIENDS_COUNT = "user_friends"
FT_PHOTO_EXISTS = "photo_exists"
FT_DAYS_SINCE_USER_CREATION = "days_since_user_creation"
FT_POS_WORDS_COUNT = "pos_words_count"
FT_NEG_WORDS_COUNT = "neg_words_count"
FT_POS_SENT = "ft_sent"


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

#Map between each tweet and a list of its replies (to be used by task B)
from collections import defaultdict
TweetRepliesMap = defaultdict(list)

def ExtractTweetFeatures(tweet,src_tweet,direct_src_tweet):
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

    tweet[FT_POS_WORDS_COUNT] = posWordsCount
    tweet[FT_NEG_WORDS_COUNT] = negWordsCount
    if posWordsCount > negWordsCount:
        tweet[SENTIMENT_WORD_FEATURE] = 1
    elif negWordsCount > posWordsCount:
        tweet[SENTIMENT_WORD_FEATURE] = 0
    else:
        tweet[SENTIMENT_WORD_FEATURE] = 0.5

    if tweet[ATTR_REPLY_STATUS_ID]:
        tweet[IS_REPLY_FEATURE] = 1
    else:
        tweet[IS_REPLY_FEATURE] = 0

    if tweet[ATTR_USER][ATTR_USER_VERIFIED] == True:
        tweet[FT_USER_VERIFIED] = 1
    else:
        tweet[FT_USER_VERIFIED] = 0

    if src_tweet[ATTR_USER][ATTR_USER_VERIFIED] == True:
        tweet[FT_SRC_USER_VERIFIED] = 1
    else:
        tweet[FT_SRC_USER_VERIFIED] = 0

    tweet[FT_USER_FOLLOWERS_COUNT] = tweet[ATTR_USER][ATTR_FOLLOWERS_COUNT]
    tweet[FT_USER_TWEETS_COUNT] = tweet[ATTR_USER][ATTR_TWEETS_COUNT]
    tweet[FT_USER_FRIENDS_COUNT] = src_tweet[ATTR_USER][ATTR_FRIENDS_COUNT]


    #Some of the direct src tweets are not found !
    if direct_src_tweet is not None and direct_src_tweet[ATTR_USER][ATTR_USER_VERIFIED] == True:
        tweet[FT_DIRECT_SRC_USER_VERIFIED] = 1
    else:
        tweet[FT_DIRECT_SRC_USER_VERIFIED] = 0

    #Compute the cosine similarity with the src tweet
    SrcTweetTokens = tknzr.tokenize(src_tweet[ATTR_TEXT])
    cosSim = get_cosine(Counter(tokens),Counter(SrcTweetTokens))
    if cosSim > 0.75:
        tweet[FT_COSINE_SIM_SRC_TWEET] = 1
        #print('%r -- %r -> %s' % (tokens,SrcTweetTokens,cosSim))
    else:
        tweet[FT_COSINE_SIM_SRC_TWEET] = 0

    tweet[FT_COSINE_SIM_SRC_DIRECT_TWEET] = 0
    if direct_src_tweet is not None:
        SrcDirectTweetTokens = tknzr.tokenize(direct_src_tweet[ATTR_TEXT])
        cosSim = get_cosine(Counter(tokens),Counter(SrcDirectTweetTokens))
        if cosSim > 0.75:
            tweet[FT_COSINE_SIM_SRC_DIRECT_TWEET] = 1

    if src_tweet[ATTR_RETWEET_COUNT] != 0:
        tweet[FT_RETWEET_RATIO] = tweet[ATTR_RETWEET_COUNT]/src_tweet[ATTR_RETWEET_COUNT]
    else:
        tweet[FT_RETWEET_RATIO] = 0
    #print('Retweet Ratio= %s' % (tweet[FT_RETWEET_RATIO]))

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

    #Compute number of days since user creation feature
    tweetTime = time.strftime(MY_DATE_FORMAT, time.strptime(tweet[ATTR_USER][ATTR_USER_CREATED_AT],TWITTER_DATE_FORMAT))
    tweetTime = datetime.strptime(tweetTime, MY_DATE_FORMAT)
    delta = currentTime - tweetTime
    tweet[FT_DAYS_SINCE_USER_CREATION] = delta.days  
    
    #Compute Sentiment
    tweet[FT_POS_SENT] = SentAnalyzer.classify(tweet[ATTR_NEW_TEXT])
    
    #to be used by task B: populate the Map between each tweet and a list of its replies
    TweetRepliesMap[src_tweet[ATTR_ID]].append(tweet)
    
    return tweet

def GetTweetsInfo (TaskData,IsTest):
    ResultTweetsList = []
    counter = 0
    for key, value in TaskData.items():       
        tweet = GetTweetObj(key,IsTest)
        if tweet is None:
            continue
            
        print(key)
        print(value)
        print(type(value))
        tweet[ATTR_CATEGORY] = le.transform([value]) #VERSION CHANGE(1): Use List of value instead of value only
        
        if IsTest:
            src_tweet_id = SrcTestTweetsDict[str(tweet[ATTR_ID])]
        else:
            src_tweet_id = SrcTweetsDict[str(tweet[ATTR_ID])]
        src_tweet = GetTweetObj(src_tweet_id,IsTest)
        direct_src_tweet_id = tweet[ATTR_REPLY_STATUS_ID]
        direct_src_tweet = GetTweetObj(str(direct_src_tweet_id),IsTest)
    
        tweet = ExtractTweetFeatures(tweet,src_tweet,direct_src_tweet)
        
        ResultTweetsList.append(tweet)
        
        counter+= 1
        
        #if counter % 100 == 0:
        #    print("%s passed" % (counter))

    return ResultTweetsList

def ExtractTestData (TweetsMap,SrcTwtsMap):
    ResultTweetsList = []
    counter = 0
    
    #print("*TweetsMap: " + str(len(TweetsMap)))
    
    for key, value in TweetsMap.items():    
        tweet = value
        
        src_tweet_id = SrcTestTweetsDict[str(tweet[ATTR_ID])]
        src_tweet = SrcTwtsMap[src_tweet_id]
        direct_src_tweet_id = tweet[ATTR_REPLY_STATUS_ID]
        
        if str(direct_src_tweet_id) in SrcTwtsMap:
            direct_src_tweet = SrcTwtsMap[str(direct_src_tweet_id)]
        elif str(direct_src_tweet_id) in TweetsMap:
            direct_src_tweet = TweetsMap[str(direct_src_tweet_id)]
        else:
            direct_src_tweet = None
        
        tweet = ExtractTweetFeatures(tweet,src_tweet,direct_src_tweet)
        
        ResultTweetsList.append(tweet)

        counter+= 1
        
        #if counter % 100 == 0:
        #    print("%s passed" % (counter))
    #print("*ResultTweetsList: " + str(len(ResultTweetsList)))
    return ResultTweetsList

# def ExtractSentiment(TweetsList,fileName):
#     TweetsSent = []
#     counter = 0
#     for tweet in TweetsList:
        
#         ErrCnt = 0
#         while True:
#             try:
#                 tweetText = tweet[ATTR_NEW_TEXT]
#                 print('hiray1')
#                 data = urllib.parse.urlencode({"text": "%r" % (tweetText) }) 
#                 #print(tweetText)
#                 u = urllib.request.urlopen("http://text-processing.com/api/sentiment/", data.encode('utf8'))
#                 the_page = u.read()
#                 print('hiray2')
#                 strJson = str(the_page)[2:-1]
#                 #print(strJson)
#                 print('hiray3')
#                 sentJSON = json.loads(strJson)
#                 sentJSON["id"] = tweet[ATTR_ID]
#                 print('hiray4')
#                 TweetsSent.append(sentJSON)
#                 print('hiray5')
#                 break
#             except Exception as e:
#                 ErrCnt += 1
#                 print ("error({0}): {1} - {2}".format(e.errno, e.strerror,ErrCnt))
            
#         counter+= 1
#         if counter % 100 == 0:
#             print("%s passed" % (counter))

#     with open(fileName, 'w') as outfile:
#         json.dump(TweetsSent, outfile)
        
TrainData = GetTweetsInfo(TaskATrainData,False)
DevData = GetTweetsInfo(TaskADevData,False)
TestData = ExtractTestData(ReplyTestTweetsMap,SrcTestTweetsMap)
GoldData = GetTweetsInfo(TaskAGoldData,True)
SourceTweetsList = ExtractTestData(SrcTestTweetsMap,SrcTestTweetsMap)
for tw in SourceTweetsList :
    TestData.append(tw)
print("Extracting features took --- %s seconds ---" % (time.time() - start_time))


# In[3]:


#Extracting data into lists for ease of access
TrainTweetsText = []
TrainTweetsCategories = []
TrainTweetsIDs = []
for item in TrainData:
    TrainTweetsIDs.append(item[ATTR_ID])
    TrainTweetsText.append(item[ATTR_NEW_TEXT])
    TrainTweetsCategories.append(item[ATTR_CATEGORY][0]) #VERSION CHANGE (result of 1): Use List of value instead of value only
    
DevTweetsText = []
DevTweetsCategories = []
DevTweetsIDs = []
for item in DevData:
    DevTweetsIDs.append(item[ATTR_ID])
    DevTweetsText.append(item[ATTR_NEW_TEXT])
    DevTweetsCategories.append(item[ATTR_CATEGORY])

TestTweetsText = []
TestTweetsIDs = []
for item in TestData:
    TestTweetsIDs.append(item[ATTR_ID])
    TestTweetsText.append(item[ATTR_NEW_TEXT])
    
GoldTweetsText = []
GoldTweetsCategories = []
GoldTweetsIDs = []
for item in GoldData:
    GoldTweetsIDs.append(item[ATTR_ID])
    GoldTweetsText.append(item[ATTR_NEW_TEXT])
    GoldTweetsCategories.append(item[ATTR_CATEGORY])


# In[167]:


#Transform the text into word vector
from nltk import word_tokenize

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

addWordVecFeatures = False

if addWordVecFeatures:
    trainFts = X_train_counts
    devFts = X_dev_counts
    testFts = X_test_counts
    goldFts = X_gold_counts


#Print stats
# print (len(TrainTweetsText))
# print (type(trainFts))
# print (trainFts.shape)
# print (devFts.shape)
# print (testFts.shape)
# print (goldFts.shape)

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


# In[168]:


#Fooling around: Print word features

#if not USE_PoS: 
    #print(count_vect.get_feature_names())
#else:
    #print(dict_vect.get_feature_names())



# In[169]:


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


# In[170]:


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

trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,QUEST_WORD_FEATURE)
trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,DENY_WORD_FEATURE)
trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,SUPPORT_WORD_FEATURE)
#No use for this
#trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,HASHTAG_EXISTS_FEATURE)

trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,URL_EXISTS_FEATURE)

trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,IS_REPLY_FEATURE)

#Sentiment features
trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,SENTIMENT_WORD_FEATURE)
#trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,FT_POS_WORDS_COUNT,True)
#trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,FT_NEG_WORDS_COUNT,True)
#trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,FT_POS_SENT)

#Accuracy is bad when including User Verified feature, but with the root or direct src users, it gets better !
#trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,FT_USER_VERIFIED)
trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,FT_SRC_USER_VERIFIED)
trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,FT_DIRECT_SRC_USER_VERIFIED)

#User's Followers count
trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,FT_USER_FOLLOWERS_COUNT,True)

#User's tweets count (not useful for act of speech recognition)
#trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,FT_USER_TWEETS_COUNT,True)

#User's friends count (not useful)
#trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,FT_USER_FRIENDS_COUNT,True)

#Cosine similarity of the sentance with the root source tweet
trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,FT_COSINE_SIM_SRC_TWEET)

#Cosine similarity of the sentance with the direct tweet it's replying to
#trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,FT_COSINE_SIM_SRC_DIRECT_TWEET)

#Retweet ratio (not useful for act of speech recognition)
#trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,FT_RETWEET_RATIO)

#Photo existance
#trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,FT_PHOTO_EXISTS)

#Days since user creation
#trainFts,devFts,testFts,goldFts = AddFeature(trainFts,devFts,testFts,goldFts,FT_DAYS_SINCE_USER_CREATION)

print (len(TrainTweetsText))
print (trainFts.shape)
print (devFts.shape)
print (testFts.shape)
print (goldFts.shape)
#print(StopWords)


# In[171]:


#Choose classifier
print('Choosing classifier')

#Better: 0.34
from sklearn.linear_model import SGDClassifier
sdgClassifier = SGDClassifier()

#Better: 0.33
from sklearn.svm import LinearSVC
linearSVCClassifier = LinearSVC()
#clf = LinearSVC(class_weight="balanced") 

# from sklearn.naive_bayes import MultinomialNB
# clf = MultinomialNB()

from sklearn.ensemble import RandomForestClassifier
randomForestClassifier = RandomForestClassifier()

from sklearn.linear_model import LogisticRegression
logisticRegressionClassifier = LogisticRegression()

from sklearn.tree import DecisionTreeClassifier
decisionTreeClassifier = DecisionTreeClassifier()

from sklearn.neural_network import MLPClassifier
mlpClassifier = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100), random_state=1)

from sklearn.ensemble import GradientBoostingClassifier
gradientBoostClassifier = GradientBoostingClassifier()

from sklearn.tree import DecisionTreeClassifier
cart = DecisionTreeClassifier()
from sklearn.ensemble import BaggingClassifier
baggingClassifier = BaggingClassifier(base_estimator=cart,n_estimators=50)

from sklearn.tree import DecisionTreeClassifier
cart = DecisionTreeClassifier()
from sklearn.ensemble import AdaBoostClassifier
adaBoostClassifier = AdaBoostClassifier(base_estimator=cart,n_estimators=100)

#Voting Classifier
from sklearn.ensemble import RandomForestClassifier
rfClf = RandomForestClassifier()
from sklearn.linear_model import LogisticRegression
lrClf = LogisticRegression()
from sklearn.neural_network import MLPClassifier
mlpClf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100), random_state=1)
from sklearn.svm import LinearSVC
svmClf = LinearSVC()
from sklearn.tree import DecisionTreeClassifier
cart = DecisionTreeClassifier()
from sklearn.ensemble import BaggingClassifier
bagClf = BaggingClassifier(base_estimator=cart,n_estimators=50)
from sklearn.tree import DecisionTreeClassifier
cart = DecisionTreeClassifier()
from sklearn.ensemble import AdaBoostClassifier
adaClf = AdaBoostClassifier(base_estimator=cart,n_estimators=100)
from sklearn.ensemble import GradientBoostingClassifier
grClf = GradientBoostingClassifier()
from sklearn.ensemble import VotingClassifier
votingClassifier = VotingClassifier(estimators=[
         ('rfClf', rfClf),('grClf',grClf), ('lrClf', lrClf), ('mlpClf', mlpClf),('svmClf',svmClf),('bagClf',bagClf),('adaClf',adaClf) ],
         voting='hard')

#Choose current classifier
clf = votingClassifier

print('Chose classifier')


# In[172]:


#Cross Validation using train data

'''
Quoted from SemEval:
The evaluation of the SDQC task needs to be more careful, as the distribution of the categories is clearly skewed 
towards comments. Given that comments are the least helpful type of message towards establishing the veracity of a 
rumour and the most populous, the evaluation metric needs to reward systems that perform well when classifying support, 
denial, and querying. This will be achieved through macroaveraged F1 aggregated for support, denial, and querying, 
disregarding the performance on comments. Individual S, D, Q and C scores will be given in the final report.
'''

cross_validate = 0

if cross_validate == 1:
    folds = 10

    resultsArr = np.array(list(TrainTweetsCategories), dtype=int)

    scorer = make_scorer(fbeta_score,greater_is_better=True,labels=[ID_DENY,ID_QUERY,ID_SUPPORT],average="macro",beta=2)

    scores = cross_validation.cross_val_score(clf, trainFts, resultsArr, cv=folds,scoring=scorer)

    print(scores)
    print("Macro Averaged F1 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    scorer = make_scorer(fbeta_score,greater_is_better=True,labels=[ID_DENY,ID_QUERY,ID_SUPPORT],average="weighted",beta=2)

    scores = cross_validation.cross_val_score(clf, trainFts, resultsArr, cv=folds,scoring=scorer)
    print(scores)
    print("Weighted Averaged F1 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    scores = cross_validation.cross_val_score(clf, trainFts, resultsArr, cv=folds)
    print(scores)
    print("Normal Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[173]:


import sys
sys.version
sys.version_info

import sklearn
import nltk
print('The nltk version is {}.'.format(nltk.__version__))
print('The scikit-learn version is {}.'.format(sklearn.__version__))


# In[174]:



def testWithDevData(clfName,classifier):

    #Train Classifier
    import numpy as np
    array = np.array(list(TrainTweetsCategories), dtype=int)

    classifier.fit(trainFts,array)

    #Test Classifier
    predicted = classifier.predict(devFts)
    #for doc, category in zip(DevData.keys(), predicted):
    #    print('%r => %s' % (doc, category))


    #Evaluate Classifier using dev data

    resultsArr = np.array(list(DevTweetsCategories), dtype=int)
    print (clfName + " Accuracy: " + str(np.mean(predicted == resultsArr)))

    print("---------------------------------------------------------------------------Without Comments")

    print(metrics.classification_report(resultsArr, predicted,target_names=target_names[1:],labels=[ID_DENY,ID_QUERY,ID_SUPPORT]))

    print("*Macro F1 Average*: " + str(metrics.f1_score(resultsArr, predicted,average='macro',labels=[ID_DENY,ID_QUERY,ID_SUPPORT])))
    print("Micro F1 Average: " + str(metrics.f1_score(resultsArr, predicted,average='micro',labels=[ID_DENY,ID_QUERY,ID_SUPPORT])))
    print("Weighted F1 Average: " + str(metrics.f1_score(resultsArr, predicted,average='weighted',labels=[ID_DENY,ID_QUERY,ID_SUPPORT])))

    print("---------------------------------------------------------------------------Includng Comments")

    print(metrics.classification_report(resultsArr, predicted,target_names=target_names,labels=[ID_COMMENT,ID_DENY,ID_QUERY,ID_SUPPORT]))

    print("*Macro F1 Average*: " + str(metrics.f1_score(resultsArr, predicted,average='macro',labels=[ID_COMMENT,ID_DENY,ID_QUERY,ID_SUPPORT])))
    print("Micro F1 Average: " + str(metrics.f1_score(resultsArr, predicted,average='micro',labels=[ID_COMMENT,ID_DENY,ID_QUERY,ID_SUPPORT])))
    print("Weighted F1 Average: " + str(metrics.f1_score(resultsArr, predicted,average='weighted',labels=[ID_COMMENT,ID_DENY,ID_QUERY,ID_SUPPORT])))

    print("Confusion Matix") 
    print(metrics.confusion_matrix(resultsArr, predicted))

show_details = False
    
if show_details == True:
    testWithDevData('Random Forest',randomForestClassifier)
    testWithDevData('Logistic Regression',logisticRegressionClassifier)
    testWithDevData('SVM',linearSVCClassifier)
    testWithDevData('Neural Network',mlpClassifier)
    testWithDevData('Gradient Boost',gradientBoostClassifier)
    testWithDevData('Decision Tree Bagging',baggingClassifier)
    testWithDevData('AdaBoost',adaBoostClassifier)
testWithDevData('Voting',votingClassifier)


# In[175]:


# #Write dev results to file & apply scorer

# ResultDict = {}
# i = 0
# while i < len(DevTweetsIDs):
#     ResultDict[DevTweetsIDs[i]] = target_names[predicted[i]]
#     i+= 1
    
# with open(SubTaskADevResultsFilePath, 'w') as outfile:
#     json.dump(ResultDict, outfile)

# ipy = get_ipython()
# ipy.magic("run ScorerA.py \"" + SubTaskADevFilePath + "\" \"" + SubTaskADevResultsFilePath + "\"")


# In[176]:


# print(type(devFts))
TrainAndDevCategories = TrainTweetsCategories+DevTweetsCategories
TrainAndDevFts = sp.vstack((trainFts,devFts),format='csr')
# print(trainFts.shape)
# print(devFts.shape)
# print(TrainAndDevFts.shape)

CategoriesArr = np.array(list(TrainAndDevCategories), dtype=int)

clf.fit(TrainAndDevFts,CategoriesArr)


# In[177]:


# #Write test results to file

print(testFts.shape)
print(len(TestTweetsIDs))

predicted = clf.predict(testFts)

ResultDict = {}
i = 0
while i < len(TestTweetsIDs):
    ResultDict[TestTweetsIDs[i]] = target_names[predicted[i]]
    TestData[i][ATTR_CATEGORY] = predicted[i]
    i+= 1
    
with open(SubTaskATestResultsFilePath, 'w') as outfile:
    json.dump(ResultDict, outfile)


# In[178]:


#Write gold results to file & apply scorer

get_ipython().magic('run getScoreA.py')

def getGoldResultsScoreA(name,classifier):

    classifier.fit(TrainAndDevFts,CategoriesArr)
    
    predicted = classifier.predict(goldFts)

    ResultDict = {}
    i = 0
    while i < len(GoldTweetsIDs):
        ResultDict[GoldTweetsIDs[i]] = target_names[predicted[i]]
        i+= 1

    with open(SubTaskAGoldResultsFilePath, 'w') as outfile:
        json.dump(ResultDict, outfile)

    # ipy = get_ipython()
    # ipy.magic("run ScorerA.py \"" + SubTaskAGoldFilePath + "\" \"" + SubTaskAGoldResultsFilePath + "\"")


    print( name + ' score: ' + str(getScoreA(SubTaskAGoldFilePath,SubTaskAGoldResultsFilePath)))

show_details = True
    
if show_details == True:
    getGoldResultsScoreA('Random Forest',randomForestClassifier)
    getGoldResultsScoreA('Logistic Regression',logisticRegressionClassifier)
    getGoldResultsScoreA('SVM',linearSVCClassifier)
    getGoldResultsScoreA('Neural Network',mlpClassifier)
    getGoldResultsScoreA('Gradient Boost',gradientBoostClassifier)
    getGoldResultsScoreA('Decision Tree Bagging',baggingClassifier)
    getGoldResultsScoreA('AdaBoost',adaBoostClassifier)
getGoldResultsScoreA('Voting',votingClassifier)

# getGoldResultsScore(logisticRegressionClassifier)
# getGoldResultsScore(decisionTreeClassifier)
# getGoldResultsScore(mlpClassifier)
# getGoldResultsScore(gradientBoostClassifier)
# getGoldResultsScore(baggingClassifier)
# getGoldResultsScore(adaBoostClassifier)
# getGoldResultsScore(votingClassifier)



# In[179]:


#Fooling around

#predicted[18]
#predicted[50]
#predicted

#import numpy
#numpy.where( predicted == 2)


# In[180]:


#Fooling around

#list(DevTweetsCategories)[18]
#list(DevTweetsCategories)[50]
#DevData.values()

#print (list(DevTweetsText)[3])
#numpy.where( resultsArr == 2)


# In[181]:


#Prediction for 1 Test

#d = ["Reports of shooting at  Dammartin en Goele on route N2 north east of Paris - French media says car chase under way"]
#x = count_vect.transform(d)
#p = clf.predict(x)
#for doc, category in zip(d, p):
#    print('%r => %s' % (doc, category))

#xx = count_vect.transform(d).toarray()
#print (numpy.where( xx != 0))

#count_vect.vocabulary_
#for kk,vv in count_vect.vocabulary_.items():
    #if vv == 663:
    #    print (kk)

#230 considered questions and are queries
#469 considered questions but are not queries
#98 are queries but not considered questions
    
Data = TrainData
#Data = DevData

count = 0
for tweet in Data:
    if tweet[QUEST_WORD_FEATURE] == 1 and tweet["category"] == ID_QUERY:
        count = count + 1
# print(count)

# for tweet in Data:
#     if tweet["category"] == ID_QUERY:
#         print('[%r] %r (%s,%s) -> %s (+%s,-%s,%s)' % (tweet["id"],tweet["text"],tweet["category"],tweet[QUEST_WORD_FEATURE],tweet[SENTIMENT_WORD_FEATURE],tweet[POS_WORDS_COUNT],tweet[NEG_WORDS_COUNT],tweet[NEUT_WORDS_COUNT]))


# In[182]:


#Fooling around: Testing Tokenizing
# tweetText = "@UlrichJvV @affinity292 how many 'nazis' did leebowitz kill ?https://t.co/bZdlBn9qiF"
# tokens = tknzr.tokenize(tweetText)
# print(tokens)
# if (Question_Words_Set.intersection(tokens)):
#     print('Question')
# else:
#     print('not Question')



# In[183]:


#Support Results
# i = 0
# r = 0
# a = 0
# while i < len(DevTweetsText):
#     if DevTweetsCategories[i] == ID_SUPPORT:
#         a = a + 1
#         if predicted[i] == ID_SUPPORT:
#             r = r + 1
#     i = i + 1

# print('%s => %s' % (a,r))

# i = 0
# while i < len(DevTweetsText):
#     if DevTweetsCategories[i] == ID_SUPPORT:
#         print('[%r] %r (%s) => %s (S=%s)' % (DevTweetsIDs[i],DevTweetsText[i], DevTweetsCategories[i],predicted[i],DevData[i][SUPPORT_WORD_FEATURE] ))
#     i = i +1


# In[184]:


#Deny Results

# i = 0
# r = 0
# a = 0
# while i < len(DevTweetsText):
#     if DevTweetsCategories[i] == ID_DENY:
#         a = a + 1
#         if predicted[i] == ID_DENY:
#             r = r + 1
#     i = i + 1

# print('%s => %s' % (a,r))

# i = 0
# while i < len(DevTweetsText):
#     if DevTweetsCategories[i] == ID_DENY:
#         print('[%r] %r (%s) => %s (D=%s)' % (DevTweetsIDs[i],DevTweetsText[i], DevTweetsCategories[i],predicted[i],DevData[i][DENY_WORD_FEATURE] ))
#     i = i +1


'''
i = 0
while i < len(TrainTweetsText):
    if TrainTweetsCategories[i] == ID_DENY:
        print('[%r] %r (%s) => (D=%s)' % (TrainTweetsIDs[i],TrainTweetsText[i], TrainTweetsCategories[i],TrainData[i][DENY_WORD_FEATURE] ))
    i = i +1
'''


# In[185]:


#Query Results
# print("Query Results")
# i = 0
# r = 0
# a = 0
# while i < len(DevTweetsText):
#     if DevTweetsCategories[i] == ID_QUERY:
#         a = a + 1
#         if predicted[i] == ID_QUERY:
#             r = r + 1
#     i = i + 1

# print('%s => %s' % (a,r))

# i = 0
# while i < len(DevTweetsText):
#     if DevTweetsCategories[i] == ID_QUERY:
#         print('[%r] %r (%s) => %s (Q=%s)' % (DevTweetsIDs[i],DevTweetsText[i], DevTweetsCategories[i],predicted[i],DevData[i][QUEST_WORD_FEATURE] ))
#     i = i +1

