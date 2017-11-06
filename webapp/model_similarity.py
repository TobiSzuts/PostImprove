"""
Contains functions for determining similarity score between posts
"""

from webapp import model_functions

def TestFunction(param):

    (a, b, c, d, e, f) = model_functions.LoadParameters2()
    
    return a


def LoadSimilarityVariables() :
    """ Save variables to disk with pickle
    """
    import pickle

    base_name = 'Similarity'
    
    file_name = '{}_titles_index.p'.format(base_name)
    titles_index = pickle.load(open(file_name, 'rb'))

    file_name = '{}_posts_index.p'.format(base_name)
    posts_index = pickle.load(open(file_name, 'rb'))

    file_name = '{}_df.p'.format(base_name)
    data_sim = pickle.load(open(file_name, 'rb'))

    
    base_name = 'Model_2bins'
    file_name = '{}_post_lsi.p'.format(base_name)
    post_lsi = pickle.load(open(file_name, 'rb'))
    
    file_name = '{}_title_lsi.p'.format(base_name)
    title_lsi = pickle.load(open(file_name, 'rb'))
    
    file_name = '{}_dictionary.p'.format(base_name)
    dictionary = pickle.load(open(file_name, 'rb'))

    return (titles_index, posts_index, data_sim, post_lsi, title_lsi, dictionary)



def FindSimilarTexts(text, dictionary, lsi, sim_index, sort_output = True) :
    """ Find the index of the most similar texts in the corpus.  
        Returns a list of tuples, where each element is (index, similarity score).
    """
    
    text_tokenized = model_functions.ProcessText([text])
    #print(text_tokenized)
    text_vec = model_functions.Vectorize_text(text_tokenized, dictionary)
    text_lsi = lsi[text_vec[0]]
    results = sim_index[text_lsi]
    
    results2 = [(x, y) for (x, y) in enumerate(results)]
    if sort_output :
        results2 = sorted(results2, key=lambda x: x[1], reverse = True)
        
    return results2


def FindSimilarPosts(post_title, post_text) :
    """ Find 5 most similar posts by taking a simple average of the title and
        post similarity, and taking the top one
    """

    (titles_index, posts_index, data_sim, post_lsi, title_lsi,
          dictionary) =  LoadSimilarityVariables()
    
    title_best = FindSimilarTexts(post_title, dictionary, title_lsi, titles_index, sort_output = False)
    post_best = FindSimilarTexts(post_text, dictionary, post_lsi, posts_index, sort_output = False)
    
    compound_best = [(i, (x+2*y)/3) for ((i, x), (j, y)) in zip(title_best, post_best)]
    compound_best = sorted(compound_best, key = lambda x: x[1], reverse = True )

    results = []
    for x in range(5) :
        index = compound_best[x][0]
        results.append([data_sim.title.iloc[index], data_sim.selftext.iloc[index],
                        data_sim.id.iloc[index], data_sim.num_comments.iloc[index]])

    print(compound_best[0])
    return results

