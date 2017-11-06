"""
This file contains functions needed for PostImprove
"""



def ProcessPost(post_title, post_text, post_time) :
    """ Process a single post from the web app.
    """
    import time
    import pandas as pd
    
    date = time.strptime(post_time, r"%Y-%m-%d %H:%M:%S")
    data = pd.DataFrame({
        'title': post_title, 
        'selftext': post_text, 
        'created_dayofweek': date.tm_wday,
        'created_hour': date.tm_hour,
        'created_month': date.tm_mon,
        'created_year': 2014,
        }, index=[0])
    
    data = CreateFeatures1(data)
    # def NLP_process(df, dictionary = None, post_lsi = None, title_lsi = None, num_lsi_topics = None, use_timer = True) :

    (num_lsi_topics, post_lsi, title_lsi, dictionary, 
         comment_bins, model) = LoadParameters2()

    (data, del1, del2, del3) = NLP_process(data, dictionary, post_lsi, title_lsi, post_lsi.num_topics, use_timer = False )

    # predict
    prediction = model.predict(data
    )[0]
    
    return prediction


# old
def ClassifyPost(title, post, time_stamp) :
    """Takes post information and classifies it according to the model.
    """
    
    (num_lsi_topics, post_lsi, title_lsi, all_dictionary, 
         comment_bins, model) = LoadParameters2()

    df = PreProcessData(title, post, time_stamp)
    df = ProcessText(df, post, title, all_dictionary, post_lsi, title_lsi)
    
    #print(df.head())
    
    # model.predict(df)
    prediction = model.predict(df)[0]
    valid_bins = ['0', '1', '2-3', '4+']
    user_text = ['Try the suggestions below', 'Look at the following suggestions', 'Not bad', 'No comments']
    response = 'This post will have {} comments'.format(valid_bins[prediction])
    response += '   ' + user_text[prediction] + '.'
    return response


def TestFunction(file_name = 'Model_binned_1_logreg_model.p') :
    # for debugging
    
    import pickle
    try :
        temp = pickle.load(open(file_name, 'rb'))
        temp = temp.C
    except :
        print("Unexpected error")
        temp = -22
#    temp = pickle.load(open(filename, 'rb'))
    
    print(temp)

    return temp


##################### Sub-functions
def LoadParameters2() :
    "quickly load in variables from disk"

    import pickle
    
    #    base_name = 'Model_binned_1'
    base_name = 'Model_2bins_all'
    
    file_name = '{}_num_lsi_topics.p'.format(base_name)
    num_lsi_topics = pickle.load(open(file_name, 'rb'))
    
    file_name = '{}_post_lsi.p'.format(base_name)
    post_lsi = pickle.load(open(file_name, 'rb'))
    
    file_name = '{}_title_lsi.p'.format(base_name)
    title_lsi = pickle.load(open(file_name, 'rb'))
    
    file_name = '{}_dictionary.p'.format(base_name)
    dictionary = pickle.load(open(file_name, 'rb'))
    
    file_name = '{}_comment_bins.p'.format(base_name)
    comment_bins = pickle.load(open(file_name, 'rb'))

    model_file_name = '{}_model.p'.format(base_name)
    model = pickle.load(open(model_file_name, 'rb'))
    
    return (num_lsi_topics, post_lsi, title_lsi, dictionary, 
         comment_bins, model)


def ProcessText(raw_text) :
    """ Destem, tokenize, and remove stop words from texts.  Raw_text should
        be an array of documents.
    """
    from nltk.tokenize import WordPunctTokenizer
    import string
    from nltk.stem.snowball import SnowballStemmer
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    word_punct_tokenizer = WordPunctTokenizer()
    stemmer = SnowballStemmer("english", ignore_stopwords=True)

    texts_proc = []
    for text in raw_text :
        words = word_punct_tokenizer.tokenize(text)
        words = [w for w in words if (w not in stop_words)]
        texts_proc.append([stemmer.stem(w.lower()) for w in words 
                                if not all(c in punctuation for c in w)])
    return texts_proc


def CreateCorpusDictionary(texts) :
    """ Create a from a tokenized text (an array of tokenized documents) 
        a corpus dictionary.
    """
    from gensim import corpora

    # create dictionary
    corpus_dictionary = corpora.Dictionary(texts)
    corpus_dictionary.filter_extremes(no_below=3, no_above=0.5)
    print(corpus_dictionary)
    return corpus_dictionary


def Vectorize_text(texts, dictionary) :
    "From a list of tokenized texts, create their vectorized representation"
    return [dictionary.doc2bow(text) for text in texts]


def ComputeDocumentLSIs(documents, lsi, N, label_base = 'lsi') : 
    " Compute the LSI representation of every document in the corpus"
    import pandas as pd
    
    baseline = [0 for x in range(N)]
    col_labels = ['{}_{}'.format(label_base, x) for x in range(N)]
    new_features = pd.DataFrame([], columns=col_labels)
    for (x, text) in enumerate(documents) :
        if len(text) > 0:
            lsi_temp = lsi[text]
            if len(lsi_temp) == N :
                temp = [y for (z,y) in lsi_temp]
                new_features.loc[x] = temp
            else :
                #print(x, len(temp))
                new_features.loc[x] = baseline
        else :
            new_features.loc[x] = baseline
    return new_features


def NLP_process(df, dictionary = None, post_lsi = None, title_lsi = None, num_lsi_topics = None, use_timer = True) :
    """ Function for NLP pre-processing.  If dictionary isn't specified, 
        create it from the posts and titles.  If post_lsi and title_lsi are not
        specified, create them as well.
    """
    from gensim.models import lsimodel
    
    if use_timer :
        my_timer = SimpleTimer()
    posts_tokenized = ProcessText(df.selftext)
#    posts_tokenized = []
    if use_timer :
        my_timer.elapsed('Processed Posts')
    
    titles_tokenized = ProcessText(df.title)
    if use_timer :
        my_timer.elapsed('Processed Titles')
    
    if not dictionary :
        dictionary = CreateCorpusDictionary(posts_tokenized + titles_tokenized)
        if use_timer :
            my_timer.elapsed('Created Dictionary')
        
    posts_vec = Vectorize_text(posts_tokenized, dictionary)
    titles_vec = Vectorize_text(titles_tokenized, dictionary)
    print(len(titles_vec), df.shape)
    df_new = df.copy()
    df_new = df_new.assign(post_word_len2 = [len(post) for post in posts_vec] )
    df_new = df_new.assign(title_word_len2 = [len(post) for post in titles_vec] )
    
    df_new = df_new[sorted(df_new.columns)]
    
    if use_timer :
        my_timer.elapsed('Vectorized')

    if not post_lsi :
        post_lsi = lsimodel.LsiModel(posts_vec, num_topics = num_lsi_topics, id2word = dictionary)
    if not title_lsi :
        title_lsi = lsimodel.LsiModel(titles_vec, num_topics = num_lsi_topics, id2word = dictionary)
        my_timer.elapsed('Trained LSI')
        
    post_lsi_features = ComputeDocumentLSIs(posts_vec, post_lsi, num_lsi_topics, label_base = 'post_lsi')
    if use_timer :
        my_timer.elapsed('Computed Post LSIs')
    title_lsi_features = ComputeDocumentLSIs(titles_vec, title_lsi, num_lsi_topics, label_base = 'title_lsi')
    if use_timer :
        my_timer.elapsed('Computed Title LSIs')
    
    post_lsi_features = post_lsi_features.set_index(df_new.index)
    title_lsi_features = title_lsi_features.set_index(df_new.index)
    
    df_new = df_new.join(post_lsi_features)
    df_new = df_new.join(title_lsi_features)
    df_new = df_new.drop(['selftext', 'title'], axis=1)
    
    if use_timer :
        my_timer.elapsed('Completed {} records'.format(len(df_new)))

    return (df_new, dictionary, post_lsi, title_lsi)


class SimpleTimer() :
    def __init__(self) :
        import time
        self.start_time = time.time()
        return
        
    def elapsed(self, message = None) :
        import time
        if message :
            print("--- {:.2e} s ---   {}".format(time.time() - self.start_time, message) )
        else :
            print("--- {} s ---".format(time.time() - self.start_time) )
        return


def CreateFeatures1(raw_data, use_timer = False) :
    """ This processes the post data, and can be used as 
    a template for the upload data.
    """
    import pandas as pd
    from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
    
    if use_timer :
        my_timer = SimpleTimer()
    
    # is this training data, or web app data?  If yes, process UTC code
    if len(raw_data.columns) > 10 : 
        # some of these columns aren't being used to create features yet
        extract_keys = ['num_comments', 'created_utc', 'title', 'selftext']
        data = raw_data[sorted(extract_keys)].copy()
    
        # remove all rows without any valid text
        data = data[data.selftext.map(len) > 0]
            
        dates = pd.to_datetime(data.created_utc, unit="s")
        data['created_dayofweek'] = dates.dt.dayofweek
        data['created_hour'] = dates.dt.hour
        data['created_month'] = dates.dt.month
        data['created_year'] = dates.dt.year
        cut_off_year = 2011
        data = data[ data.created_year > cut_off_year]
        print('Removing all posts before 2011. ', end='')
        print('Earliest post = {}'.format(pd.to_datetime(data.created_utc.min(), unit="s")))
        data = data.drop('created_utc', axis=1)
    else :
        data = raw_data
    
    data['post_char_len'] = data.selftext.apply(lambda x: len(x))
    data['post_num_qs'] = data.selftext.apply(lambda x: x.count('?'))
    data['title_char_len'] = data.title.apply(lambda x: len(x))
    data['title_num_qs'] = data.title.apply(lambda x: x.count('?'))
    
    data['post_word_len1'] = data.selftext.apply(lambda x: len(x.split()) )
    data['title_word_len1'] = data.title.apply(lambda x: len(x.split()) )
    
    if use_timer :
        my_timer.elapsed('Done with simple counts')
    
    def CountPostPunctuation(row) :
        # count the number of punctuation in the selftext
        import string
        punc_set = set(string.punctuation)
        num_punc = 0
        for char in row['selftext'] :
            if char in punc_set :
                num_punc += 1
        return num_punc
    def CountTitlePunctuation(row) :
        # count the number of punctuation in the selftext
        import string
        punc_set = set(string.punctuation)
        num_punc = 0
        for char in row['title'] :
            if char in punc_set :
                num_punc += 1
        return num_punc

    data['post_num_punc'] = data.apply(CountPostPunctuation, axis=1)
    data['title_num_punc'] = data.apply(CountTitlePunctuation, axis=1)
    data['post_perc_punc'] = data.post_num_punc / data.post_char_len
    data['title_perc_punc'] = data.title_num_punc / data.title_char_len
    data.post_perc_punc = data.post_perc_punc.fillna(0)
    
    if use_timer :
        my_timer.elapsed('Done with percentage counts')
    
    # Sentiment features
    sia_ps = SIA().polarity_scores
    
    posts_sent = []
    for post in data.selftext : 
        temp = sia_ps(post)
        posts_sent.append([temp[k] for k in sorted(temp.keys())])
            
    titles_sent = []
    for title in data.title :
        temp = sia_ps(title)
        titles_sent.append([temp[k] for k in sorted(temp.keys())])
    
    data['title_compound'] = [t[0] for t in titles_sent]
    data['title_neg'] = [t[1] for t in titles_sent]
    data['title_neu'] = [t[2] for t in titles_sent]
    data['title_pos'] = [t[3] for t in titles_sent]
    
    data['post_compound'] = [t[0] for t in posts_sent]
    data['post_neg'] = [t[1] for t in posts_sent]
    data['post_neu'] = [t[2] for t in posts_sent]
    data['post_pos'] = [t[3] for t in posts_sent]
       
    if use_timer :
        my_timer.elapsed('Done')
        
    return data



################# OLD FUNCTIONS ##################

# old
def PreProcessData_old(title, post, post_time) :
    """ This processes user entered data.  All input parameters are strings.
    """
    import pandas as pd
    import time
    
    date = time.strptime(post_time, r"%Y-%m-%d %H:%M:%S")
    data = pd.DataFrame({
        'title': title, 
        'selftext': post, 
        'created_dayofweek': date.tm_wday,
        'created_hour': date.tm_hour,
        'created_month': date.tm_mon,
        'created_year': 2014,
        }, index=[0])
        
    data['post_char_len'] = data.selftext.apply(lambda x: len(x))
    data['post_num_qs'] = data.selftext.apply(lambda x: x.count('?'))
    data['title_char_len'] = data.title.apply(lambda x: len(x))
    data['title_num_qs'] = data.title.apply(lambda x: x.count('?'))
    
    def CountPostPunctuation(row) :
        # count the number of punctuation in the selftext
        import string
        punc_set = set(string.punctuation)
        num_punc = 0
        for char in row['selftext'] :
            if char in punc_set :
                num_punc += 1
        return num_punc
    def CountTitlePunctuation(row) :
        # count the number of punctuation in the selftext
        import string
        punc_set = set(string.punctuation)
        num_punc = 0
        for char in row['title'] :
            if char in punc_set :
                num_punc += 1
        return num_punc

    data['post_num_punc'] = data.apply(CountPostPunctuation, axis=1)
    data['title_num_punc'] = data.apply(CountTitlePunctuation, axis=1)
    data['post_perc_punc'] = data.post_num_punc / data.post_char_len
    data['title_perc_punc'] = data.title_num_punc / data.title_char_len
    data.post_perc_punc = data.post_perc_punc.fillna(0)
        
    return data

# old
def ProcessText_old(df, post, title, dictionary, post_lsi, title_lsi) :
    """ Process text fields to create all features.
        Returns an dataframe containing the new features.
    """
    import pandas as pd
    
    def helper(text, dictionary, lsi) :
        "process just one text"
        #from gensim import corpora, models, similarities
        from nltk.tokenize import WordPunctTokenizer
        import string
        from nltk.stem.snowball import SnowballStemmer
        from nltk.corpus import stopwords

        stop_words = set(stopwords.words('english'))
        punctuation = set(string.punctuation)
        word_punct_tokenizer = WordPunctTokenizer()
        stemmer = SnowballStemmer("english", ignore_stopwords=True)

        words = word_punct_tokenizer.tokenize(text)
        words = [w for w in words if (w not in stop_words)]
        text_tokenized = [stemmer.stem(word.lower()) for word in words]
        text_vec = dictionary.doc2bow(text_tokenized)
                
        text_features = lsi[text_vec]
        if len(text_features) == lsi.num_topics :
            text_features = [y for (z,y) in text_features]
        else :
            text_features = [0 for x in range(lsi.num_topics)]
    
        return text_tokenized, text_vec, text_features
    
    (post_tokenized, post_vec, post_features) = helper(post, dictionary, post_lsi) 
    (title_tokenized, title_vec, title_features) = helper(title, dictionary, title_lsi) 
    
    df_new = df.copy()
    df_new = df_new.assign(post_word_len1 = len(post.split()))
    df_new = df_new.assign(post_word_len2 = len(post))
    df_new = df_new.assign(title_word_len1 = len(title.split()))
    df_new = df_new.assign(title_word_len2 = len(title))
    
    post_col_names = ['post_lsi_{}'.format(x) for x in range(len(post_features))]
    post_features = pd.DataFrame(data=[post_features], columns=post_col_names, index=[0])
    df_new = df_new.join(post_features)

    title_col_names = ['title_lsi_{}'.format(x) for x in range(len(title_features))]
    title_features = pd.DataFrame(data=[title_features], columns=title_col_names, index=[0])
    df_new = df_new.join(pd.DataFrame(title_features, index=[0]))
                           
    df_new = df_new.drop(['selftext', 'title'], axis=1)
                           
    def AddLSAFeatures(data_old, post_lsi_features, title_lsi_features) :
        """ Add the LSA features to the dataframe, remove the title and post fields.
            Need to reset index values so no rows are dropped.
        """
        data_new = data_old.copy()
        post_lsi_features = post_lsi_features.set_index(data_new.index)
        title_lsi_features = title_lsi_features.set_index(data_new.index)
        data_new = data_new.join(post_lsi_features)
        data_new = data_new.join(title_lsi_features)
        data_new = data_new.drop(['selftext', 'title'], axis=1)

        return data_new
                           

    def ComputeDocumentLSIs(documents, lsi, N, label_base = 'lsi') : 
        " Compute the LSI representation of every document in the corpus"
        baseline = [0 for x in range(N)]
        col_labels = ['{}_{}'.format(label_base, x) for x in range(N)]
        new_features = pd.DataFrame([], columns=col_labels)
        for (x, text) in enumerate(documents) :
            if len(text) > 0:
                lsi_temp = lsi[text]
                if len(lsi_temp) == N :
                    temp = [y for (z,y) in lsi_temp]
                    new_features.loc[x] = temp
                else :
                    #print(x, len(temp))
                    new_features.loc[x] = baseline
            else :
                new_features.loc[x] = baseline
        return new_features

                           
    return df_new

