# PostImprove

Initial README document for the Insight Project "PostImprove: Getting the most out of an online health forum", created September 2017.

## Summary

This project was created as part of the Insight Health Data Science
Fellowship Program.  Using 90k posts to the Depression subreddit (from
Reddit.com), it creates a predictive model to classify whether a given
post will be successful (defined as having 3 or more comments) or not.
The predictive model is written using Python's sklearn module, with
natural language processing in Gensim and NLTK.  The web application
uses Flask for local debugging; in the version uploaded to AWS's EC2,
Gunicorn provides a more robust server.

## Details

The data source came from Chenhao Tan, who downloaded all of Reddit in
mid 2014 and posted it
[online](https://chenhaot.com/papers/multi-community.html).  This is a
25 GB file that uncompresses to 141 GB.  From this I extracted all
posts in the Depression Subreddit.

The data was processed by compiling various word and punctuation
statistics (word length, character length, number of question marks,
etc.), clustering into 50 topics (Latent Semantic Analysis), and
analyzing text sentiment (VADER).  In addition, text similarity was
computed by cosine distance using the 50 LSA topics.  This allows the
webapp to recommend similar posts that had 5 or more comments.

The predictive model is a Random Forest, optimized by random grid search.

## Organization

The jupyter notebooks contain my development notes and are written
chronologically.  The pickled python variables called by the web app
(the fitted model, post information, LSA representation, etc.) are not
included here for both size and security reasons.



Tobi Szuts, October 2017
