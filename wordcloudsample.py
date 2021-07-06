import pickle
import pandas as pd
import webbrowser
import dash
import dash_html_components as html
import dash_core_components as dcc

from dash.dependencies import Input, Output ,State
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objects as go 
from wordcloud import WordCloud
import matplotlib.pyplot as plt



#text = open('review.txt', mode='r',encoding='utf-8').read()

#stopwords=STOPWORDS
#wc = WordCloud(
   #background_color='white',
    #stopwords=stopwords,
    #height=600,
    #width=400
  #)

#wc.generate(text)
#wc.to_file('wordc_loud_sample_output.png')
data=pd.read_csv('balanced_reviews.csv')
data.isnull().sum() 
data[data.isnull().any(axis=1)] 
data.dropna(inplace=True) 
data = data[data['overall']!=3] 
data['Positivity'] = np.where(data['overall'] > 3, 1, 0) 

labels = ["Positive","Negative"]
values=[len(data[data.Positivity == 1]), len(data[data.Positivity == 0])]
scrape = pd.read_csv(r"scrappedReviews.csv")
scrape.head()







text = open('review.txt', mode='r',encoding='utf-8').read()

x2011 = data["reviewText"][data["Positivity"]==1]
x2012 = data["reviewText"][data["Positivity"]==0]

plt.subplots(figsize = (8,8))

wordcloud1 = WordCloud (
                    background_color = 'white',
                    width = 200,
                    height = 284
                        ).generate(' '.join(x2011))
fig1=plt.imshow(wordcloud1) 
fig1=plt.axis('off') 
plt.savefig('assets/Plotly-World_Cloudpos.png')
wordcloud2 = WordCloud (
                    background_color = 'white',
                    width = 200,
                    height = 284
                        ).generate(' '.join(x2012))
fig1=plt.imshow(wordcloud2) 
fig1=plt.axis('off')
plt.savefig('assets/Plotly-World_Cloudneg.png')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
