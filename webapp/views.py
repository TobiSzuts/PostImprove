from flask import render_template
from flask import request
from webapp import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
from webapp import a_Model
from webapp import model_functions
from webapp import model_similarity
#import ModelItb


user = 'tobiszuts' #add your username here (same as previous postgreSQL)            
host = 'localhost'
dbname = 'birth_db'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user)


#@app.route('/')
#@app.route('/index')
def index():
    return render_template("index.html",
       title = 'Home', user = { 'nickname': 'Miguel' },
       )

#@app.route('/db')
def birth_page():
    sql_query = """                                                             
                SELECT * FROM birth_data_table WHERE delivery_method='Cesarean'\
;                                                                               
                """
    query_results = pd.read_sql_query(sql_query,con)
    births = ""
    print( query_results[:10] )
    for i in range(0,10):
        births += query_results.iloc[i]['birth_month']
        births += "<br>"
    return births

#@app.route('/db_fancy')
def cesareans_page_fancy():
    sql_query = """
               SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean';
                """
    query_results=pd.read_sql_query(sql_query,con)
    births = []
    for i in range(0,query_results.shape[0]):
        births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
    return render_template('cesareans.html',births=births)
 
#@app.route('/input_old')
def cesareans_input():
    return render_template("input.html")

@app.route('/')
@app.route('/index')
@app.route('/input')
def postimprove_input():
    return render_template("postimprove_input.html")


"""
@app.route('/output')
def cesareans_output():
    return render_template("output.html")
"""

"""  for using without a model
@app.route('/output')
def cesareans_output():
  #pull 'birth_month' from input field and store it
  patient = request.args.get('birth_month')
    #just select the Cesareans  from the birth dtabase for the month that the user inputs
  query = "SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean' AND birth_month='%s'" % patient
  print(query)
  query_results=pd.read_sql_query(query,con)
  print(query_results)
  births = []
  for i in range(0,query_results.shape[0]):
      births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
      the_result = ''
  return render_template("output.html", births = births, the_result = the_result)
"""


#@app.route('/output_old')
def cesareans_output():
    #pull 'post_title' from input field and store it
    patient = request.args.get('post_title')
    #just select the Cesareans  from the birth dtabase for the month that the user inputs
    if len(patient) > 3 :
        return render_template("output.html", births = [], the_result = 'Invalid input!')
    query = "SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean' AND birth_month='%s'" % patient
    print(query)
    query_results=pd.read_sql_query(query,con)
    print(query_results)
    births = []
    for i in range(0,query_results.shape[0]):
        births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
        the_result = a_Model.ModelIt(patient,births)
    return render_template("output.html", births = births, the_result = the_result)

@app.route('/output')
def postimprove_output():
    #pull input fields and store as a variable
    post_title = request.args.get('post_title')
    post_text = request.args.get('post_text')
    post_time = request.args.get('created_utc')

    from datetime import datetime
    if not post_time :
        post_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    prediction = model_functions.ProcessPost(post_title, post_text, post_time)
    user_text = ['Nice post, but you might be able to get more responses!', 'Good post!']
    similar_text = ['Check out how these posts were written:', 'Similar posts']
    prediction_text = user_text[prediction]

    best_matches = model_similarity.FindSimilarPosts(post_title, post_text)

    urls = {}
    for match in best_matches :
        urls[match[0]] = (match[1],
                          'https://www.reddit.com/r/depression/comments/{}/'.format(match[2]),
                          match[3] )
                                                                    
    
    return render_template("postimprove_output.html",
                           post_title_entered = '"'+post_title+'"',
                           post_text_entered = post_text,
                           post_time_entered = post_time,
                           prediction = user_text[prediction],
                           urls = urls,
                           similar_text = similar_text[prediction])
