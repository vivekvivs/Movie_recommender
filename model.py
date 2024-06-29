import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert_other(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if(counter!=3):
         L.append(i['name'])
         counter+=1
        else:
            break
    return L

def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L

def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list= sorted(list(enumerate(distances)),reverse=True, key=lambda x:x[1])[1:6]

    for i in movie_list:
        print(new_df.iloc[i[0]].title)

ps=PorterStemmer()

movies = pd.read_csv("/Users/vivekjain/Documents/MLProjects/Movies Recommendor/tmdb_5000_movies.csv")
credits= pd.read_csv("/Users/vivekjain/Documents/MLProjects/Movies Recommendor/tmdb_5000_credits.csv")
#print(movies.columns)
movies = movies.merge(credits,on='title')
#print(movies)
#print(movies.columns)
movies= movies[['movie_id','title','overview','genres','keywords','cast','crew']]
#print(movies.head())
movies.dropna(inplace=True)
#print(movies.isnull().sum())
#print(movies.duplicated().sum())

#print(movies.iloc[0].genres)

movies['genres']=movies['genres'].apply(convert)
#print(movies['genres'])
movies['keywords']=movies['keywords'].apply(convert)
movies['cast']=movies['cast'].apply(convert_other)
movies['crew']=movies['crew'].apply(fetch_director)
#print(movies['crew'])
movies['overview']=movies['overview'].apply(lambda x:x.split())
#print(movies['overview'])

movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']
new_df = movies[['movie_id','title','tags']]
#print(new_df.head())

new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))
#print(new_df['tags'][0])
new_df['tags']=new_df['tags'].apply(lambda x: x.lower())
#print(new_df['tags'][0])
new_df['tags']=new_df['tags'].apply(stem)

cv = CountVectorizer(max_features=5000,stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
#print(cv.get_feature_names_out())

similarity=cosine_similarity(vectors)

pickle.dump(new_df,open('movies.pkl','wb'))
pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))






