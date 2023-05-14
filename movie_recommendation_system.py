#!/usr/bin/env python
# coding: utf-8

# # MOVIE   RECOMMENDATION  SYSTEM

# In[11]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


movies_df = pd.read_csv('movies.csv')


movies_df['year'] = movies_df['title'].str.extract('\((\d{4})\)', expand=False)
movies_df['genres'] = movies_df['genres'].str.split('|')


movies_df['genres'] = movies_df['genres'].apply(lambda x: ' '.join(x))
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['genres'])


cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


def get_recommendations(title, cosine_sim=cosine_sim, movies_df=movies_df, top_n=10):
    
    idx = movies_df[movies_df['title'] == title].index[0]

    
    sim_scores = list(enumerate(cosine_sim[idx]))

    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices].tolist()


# In[13]:


get_recommendations('Balto (1995)')




# In[3]:


import os 


# In[6]:


os.getcwd()


# In[5]:


os.chdir("C:\\Users\\ronit\\Downloads\\Movie-Recommendation-System-Dataset-main")


# In[ ]:




