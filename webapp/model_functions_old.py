"""
This file contains functions needed for PostImprove
"""


def LoadParameters2() :
    "quickly load in variables from disk"

    import pickle
    
    base_name = 'Model_binned_1'
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

    model_file_name = 'Model_binned_1_logreg_model.p'
    model = pickle.load(open(model_file_name, 'rb'))
    
    return (num_lsi_topics, post_lsi, title_lsi, dictionary, 
         comment_bins, model)


def PreProcessData(title, post, post_time) :
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


def ProcessText(df, post, title, dictionary, post_lsi, title_lsi) :
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
