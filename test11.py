#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import ast
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances


# In[2]:


credits=pd.read_csv("credits.csv")
def extract_actors(obj):
    L=[]
    count=0
    for a in ast.literal_eval(obj):
        if count!=3:
            L.append(a["name"])
            count+=1
        else:
            break
    return L
credits["cast"]=credits["cast"].apply(extract_actors)
def extract_directer(obj):
    for a in ast.literal_eval(obj):
        if a["job"]=="Director":
            return a["name"]
credits["directer"]=credits["crew"].apply(extract_directer)
credits


# In[3]:


credits["cast"]=credits["cast"].apply(lambda x:[a.replace(" ","") for a in x])
credits["cast"]=credits["cast"].apply(lambda x:[a.lower() for a in x])
credits["directer"]=credits["directer"].astype(str)
credits["directer"]=credits["directer"].apply(lambda x:x.lower())
credits["directer"]=credits["directer"].apply(lambda x:x.replace(" ",""))
credits=credits[["id","cast","directer"]]
credits


# In[4]:


keywords=pd.read_csv("keywords.csv")
def extracter(obj):
    L=[]
    for a in ast.literal_eval(obj):
        L.append(a['name'])
    return L
keywords["keywords"]=keywords["keywords"].apply(extracter)
keywords=credits.merge(keywords,on="id")
keywords


# In[5]:


movies=pd.read_csv("movies_metadata.csv")
movies=movies[["id","original_title","overview","genres","original_language","popularity",
               "vote_average","vote_count","release_date","title"]]


# In[46]:


movies["genres"]=movies["genres"].apply(extracter)
movies["genres"]=movies["genres"].apply(lambda x:[a.lower() for a in x])
#print(movies["id"].index[movies["id"]=="2014-01-01"].tolist())
movies=movies.drop(35587)
#print(movies["id"].index[movies["id"]=="1997-08-20"].tolist())
movies=movies.drop(19730)
#print(movies["id"].index[movies["id"]=="2012-09-29"].tolist())
movies=movies.drop(29503)
movies["id"]=movies["id"].astype(int)
movies=movies.merge(keywords,on="id")

movies["overview"]=movies["overview"].astype(str)
movies["overview"]=movies["overview"].apply(lambda x:x.split())
movies["overview"]=movies["overview"].apply(lambda x:[a.lower() for a in x])



movies["original_title"]=movies["original_title"].astype(str)
movies["original_title"]=movies["original_title"].apply(lambda x:x.replace(" ",""))
movies["original_title"]=movies["original_title"].apply(lambda x:x.lower())


# In[7]:


# import json
# def extract_name(str):
#     try:
#         dict = json.loads(str.replace("'", '"'))
#         return dict.get('name', 'No Name')
#     except (json.JSONDecodeError, AttributeError):
#         return ''
# movies['belongs_to_collection'] = movies['belongs_to_collection'].apply(extract_name)
# movies['belongs_to_collection']=movies['belongs_to_collection'].apply(lambda x:x.replace(" ",""))
# movies['belongs_to_collection']=movies['belongs_to_collection'].apply(lambda x:x.lower())


# In[8]:


movies


# In[9]:


movies["tags"]=movies["genres"]+movies["cast"]+movies["overview"]+movies["keywords"]
titles_to_delete=["Luv","Sur","Lilla Jönssonligan på styva linan"]
movies=movies[~movies['title'].isin(titles_to_delete)]
movies


# In[10]:


content=movies[["id","original_title","tags","directer"]]
content["tags"]=content["tags"].apply(lambda x:" ".join(x))
#content["tags"]=content["tags"].apply(lambda x:x.replace(" ,"," "))
# content["tags"]=content["tags"].apply(lambda x:x.lower())
content=content[:42500]
# for x in content["overview"]:
#     print(type(x))
content["tags"]=content["original_title"]+" "+content["directer"]+" "+content["tags"]
content.loc[0]


# In[11]:


'''from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(max_features=10000,stop_words="english")
count_matrix = count_vectorizer.fit_transform(content["tags"])
count_array = count_matrix.toarray()
feature_names = count_vectorizer.get_feature_names_out()
count_df = pd.DataFrame(count_array, columns=feature_names)
print(count_df)'''




from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=10000,stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(content["tags"])
tfidf_array = tfidf_matrix.toarray()
feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(tfidf_array, columns=feature_names)
print(tfidf_matrix.shape)


ids=movies[["id"]][:42500]
tfidf_df=ids.merge(tfidf_df,left_index=True, right_index=True)
# tfids_df=tfidf_df.set_index(ids)
tfidf_df


# In[12]:


# def centroid(obj):
#     title_to_id=dict(zip(movies['original_title'], movies["id"]))
#     ids=[title_to_id[title] for title in obj]
#     centroid=tfidf_df.loc[tfidf_df['id'].isin(ids)].mean()
#     return centroid
    
# L=["ironman","ironman2"]
# centroid_vector=centroid(L)
# centroid_vector
# centroid_vector=pd.DataFrame([centroid_vector], columns=tfidf_df.columns)
# tfidf_df=pd.concat([tfidf_df,centroid_vector], ignore_index=True)
# tfidf_df


# In[15]:


# from sklearn.metrics.pairwise import euclidean_distances
# query_vector = tfidf_df.iloc[-1, 1:].values.reshape(1, -1)
# other_vectors = tfidf_df.iloc[:-1, 1:].values
# distances = euclidean_distances(query_vector, other_vectors)
# k=100
# # min_distance = distances.min()  # Get the minimum value from the distances array
# # print(min_distance)
# # max_distance = distances.max()  # Get the minimum value from the distances array
# # print(max_distance)
# k_indices = np.argsort(distances)[0][:k]
# # print(f"Indices of nearest neighbors: {k_indices}")
# recs=tfidf_df["id"][k_indices]
# # print(recs)


# In[16]:


# movids=movies.loc[movies['id'].isin(recs)]
# movids


# In[17]:


# title_to_id=dict(zip(movies['id'], movies['title']))
# recommendations_title_id=[title_to_id[id] for id in recs]
# (recommendations_title_id)


# In[18]:


# from sklearn.metrics.pairwise import cosine_similarity
# last_row = tfidf_df.iloc[-1, 1:].values.reshape(1, -1)
# other_rows = tfidf_df.iloc[:-1, 1:].values
# similarities = cosine_similarity(last_row, other_rows)
# one=1
# similarities=np.append(similarities,one)
# similarities
# tfidf_df['cosine_similarity'] = similarities[0]

# #Display the DataFrame without the 'id' column
# result_df = tfidf_df.drop(columns=['id'])
# print(result_df)

#siims=result_df["cosine_similarity"]
# result_df=result_df.sort_values(by='cosine_similarity', ascending=False)
# result_df


# In[19]:


# from sklearn.metrics.pairwise import cosine_similarity
# cosine_similarities=cosine_similarity(tfidf_df.loc[42500],tfidf_df)
# cosine_similarities


# In[20]:


# import numpy as np
# from collections import Counter
# def centroid(obj):
#     title_to_id=dict(zip(movies['original_title'], movies["id"]))
#     ids=[title_to_id[title] for title in obj]
#     centroid=tfidf_df.loc[tfidf_df['id'].isin(ids)].mean()
#     return centroid
    
# L = ["ironman", "batman", "theavengers"]
# centroid_vector = centroid(L)
# centroid_vector

# class KNN:
#     def __init__(self, k=3):
#         self.k = k
        
#     def fit(self, X):
#         self.X_train = X
        
#     def predict(self, X):
#         y_pred = [self._predict(x) for x in X]
#         return np.array(y_pred)
    
#     def _predict(self, x):
#         distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
#         k_indices = np.argsort(distances)[1:self.k+1]
#         return k_indices

# if __name__ == "__main__":
#     X =   centroid_vector # Get data from the function
    
#     clf = KNN(k=2)
#     clf.fit(X)
    
#     nearest_neighbors = clf.predict(X)
    
#     for point, neighbors in zip(X, nearest_neighbors):
#         print(f"Point {point} has nearest neighbors: {X[neighbors]}")


# In[21]:


# from sklearn.metrics.pairwise import cosine_similarity
# cosine_similarities=cosine_similarity(tfidf_matrix,tfidf_matrix)
# cosine_similarities


# In[22]:


#  def get_recommendations_title(title,movies):
#     if title=="skip":
#          return None
#     index = movies[movies["original_title"]==title].index[0]
#     if index==0:
#         print("Movie not found1")
#         return None
#     sim_scores = list(enumerate(cosine_similarities[index]))
#     if not sim_scores:
#         print("Movie not found2")
#         return None
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     sim_scores = sim_scores[1:11]
#     movie_indices = [score[0] for score in sim_scores]
#     return movies['id'].iloc[movie_indices]

# movie="ironman"
# recommendations_title = get_recommendations_title(movie,movies)
# print(f"Recommended movies for {movie}:")
# print(recommendations_title)


# In[23]:


# def list_genre(genre,movies):
#     L = []
#     for i, x in enumerate(movies["genres"]):
#         if genre in x:
#             L.append(i)
#     return L

# def get_recommendations_genre(genre,movies):
#     if genre=="skip":
#         return None
#     L = list_genre(genre,movies)
#     if not L:
#         print("Movie not found")
#         return None
#     return movies['title'].iloc[L]

# movie_genre = 'foreign'
# recommendations_genre = get_recommendations_genre(movie_genre,movies)
# print(f"Recommended movies for {movie_genre}:")
# print(recommendations_genre)


# In[24]:


# def list_cast(cast,movies):
#     L = []
#     for i, x in enumerate(movies["cast"]):
#         if cast in x:
#             L.append(i)
#     return L
    
# def get_recommendations_cast(cast,movies):
#     if cast=="skip":
#         return None
#     L = list_cast(cast,movies)
#     if not L:
#         print("Movie not found")
#         return None
#     return movies['id'].iloc[L]
    
# # movie_cast = 'tomcruise'
# # recommendations_cast = get_recommendations_cast(movie_cast)
# # print(f"Recommended movies for {movie_cast}:")
# # print(recommendations_cast)


# In[25]:


# def list_lang(lang,movies):
#     L = []
#     for i, x in enumerate(movies["original_language"]):
#         if lang==x:
#             L.append(i)
#     return L

# def get_recommendations_lang(lang,movies):
#     if lang=="skip":
#         return None
#     L = list_lang(lang)
#     if not L:
#         print("Movie not found")
#         return None
#     return movies['id'].iloc[L]

# # movie_lang = 'en'
# # recommendations_lang = get_recommendations_lang(movie_lang,movies["original_language"])
# # print(f"Recommended movies for {movie_lang}:")
# # print(recommendations_lang)


# In[26]:


# def list_dir(dir,movies):
#     L = []
#     for i, x in enumerate(movies["directer"]):
#         if dir==x:
#             L.append(i)
#     return L

# def get_recommendations_dir(dir,movies):
#     L = list_dir(dir,movies)
#     if not L:
#         print("Movie not found")
#         return None
#     return movies['id'].iloc[L]
    
# # movie_dir = 'christophernolan'
# # recommendations_dir = get_recommendations_dir(movie_dir)
# # print(f"Recommended movies for {movie_dir}:")
# # print(recommendations_dir)


# In[27]:


# def list_collection(collection,movies):
#     L = []
#     for i, x in enumerate(movies["belongs_to_collection"]):
#         if collection==x:
#             L.append(i)
#     return L

# def get_recommendations_collection(collection,movies):
#     L = list_dir(collection,movies)
#     if not L:
#         print("Movie not found")
#         return None
#     return movies['title'].iloc[L]

# # movie_collection = 'starwarscollection'
# # recommendations_collection = get_recommendations_dir(movie_collection)
# # print(f"Recommended movies for {movie_lang}:")
# # print(recommendations_collection)


# In[28]:


# SET BASED SYSTEM BASICALLY 3 SET KA INTERSECTION NIKALEGA
# recommendations_title_set=set(recommendations_title) 
# recommendations_genre_set=set(recommendations_genre)
# recommendations_lang_set=set(recommendations_lang)
# recommendations=recommendations_title_set.intersection(recommendations_genre_set,recommendations_lang_set)
# for x in recommendations:
#     print(x)


# In[29]:


# def popularity_sort(obj,movies):
#     populars=movies[["popularity","id"]]
#     populars=populars.loc[populars['id'].isin(obj)]
#     populars["popularity"]=populars["popularity"].astype(str)
#     populars=populars.sort_values(by="popularity", ascending=False)
#     populars=populars["id"][:10]
#     return populars
    # title_to_id=dict(zip(movies['id'], movies['title']))
    # return [title_to_id[id] for id in populars]



# In[30]:


# L=["Luv"]
# title_to_id=dict(zip(movies['title'], movies['id']))
# recommendations_title_id=[title_to_id[id] for id in L]
# (recommendations_title_id)

# recommendations_genre_id=[title_to_id[id] for id in recommendations_genre_sort]
# (recommendations_genre_id)

# recommendations_lang_id=[title_to_id[id] for id in recommendations_lang_sort]
# (recommendations_lang_id)


# In[31]:


# recs=[24428.0,2661.0,57548.0,46770.0,99861.0,300424.0,222619.0,222619.0,259910.0,10138.0,271110.0,119569.0,21683.0]
# title_to_id=dict(zip(movies['id'], movies['title']))
# recommendations_title_id=[title_to_id[id] for id in recs]
# (recommendations_title_id)


# In[32]:


# def get_recs_title(obj,movies,tfidf_df):
#     title_to_id=dict(zip(movies['original_title'], movies["id"]))
#     ids=[title_to_id[title] for title in obj]
#     centroid=tfidf_df.loc[tfidf_df['id'].isin(ids)].mean()
#     centroid_vector=pd.DataFrame([centroid], columns=tfidf_df.columns)
#     tfidf_df=pd.concat([tfidf_df,centroid_vector], ignore_index=True)
#     query_vector = tfidf_df.iloc[-1, 1:].values.reshape(1, -1)
#     other_vectors = tfidf_df.iloc[:-1, 1:].values
#     distances = euclidean_distances(query_vector, other_vectors)
#     k=100
#     k_indices = np.argsort(distances)[0][:k]
#     recs=tfidf_df["id"][k_indices]
#     return recs


# In[30]:


# def get_recs_lang(lang,movies):
#     L = []
#     for i, x in enumerate(movies["original_language"]):
#         if lang==x:
#             L.append(i)
#     if not L:
#         print("Movie not found lang")
#         return None
#     return movies['id'].iloc[L]


# In[31]:


# def get_recs_genre(genre,movies):
#     L = []
#     for i, x in enumerate(movies["genres"]):
#         if genre in x:
#             L.append(i)
#     if not L:
#         print("Movie not found genre")
#         return None
#     return movies['title'].iloc[L]


# In[32]:


# def get_recs_dir(dir,movies):
#     L = []
#     for i, x in enumerate(movies["directer"]):
#         if dir==x:
#             L.append(i)
#     if not L:
#         print("Movie not found dir")
#         return None
#     return movies['id'].iloc[L]


# In[33]:


# def get_recs_actor(cast,movies):
#     L = []
#     for i, x in enumerate(movies["cast"]):
#         if cast in x:
#             L.append(i)
#     if not L:
#         print("Movie not found actor")
#         return None
#     return movies['id'].iloc[L]


# In[12]:


ratings = pd.read_csv('ratings_small.csv')
movie_id_to_title = dict(zip(movies['id'].astype(str), movies['title']))
# movie_title_to_id = {v: k for k, v in movie_id_to_title.items()}
movie_title_to_id = dict(zip(movies['original_title'].astype(str), movies['id']))
ratings = ratings.drop('timestamp', axis=1)
ratings['rating'].fillna(0, inplace=True)
ratings = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
ratings.columns = ratings.columns.astype(str)


# In[41]:


def get_recs(titles,lang,a,combo,movies,tfidf_df):

    lang_to_lag={"English":"en","French":"fr","Italian":"it","Japanese":"ja","German":"de","Spanish":"es","Russian":"ru","Hindi":"hi","Korean":"ko","Chinese":"zh","skip":"skip"}
    lang=lang_to_lag[lang]

    combo=combo.replace(" ","").lower()


    
    titles=titles.split(',')
    titles = [x.replace(" ", "").lower() for x in titles]
    if len(titles)==1:
        return []
        
    # rec_ids=get_recs_title(titles,movies,tfidf_df)
    title_to_id=dict(zip(movies['original_title'], movies["id"]))
    ids=[title_to_id[title] for title in titles]
    centroid=tfidf_df.loc[tfidf_df['id'].isin(ids)].mean()
    centroid_vector=pd.DataFrame([centroid], columns=tfidf_df.columns)
    tfidf_df=pd.concat([tfidf_df,centroid_vector], ignore_index=True)
    query_vector = tfidf_df.iloc[-1, 1:].values.reshape(1, -1)
    other_vectors = tfidf_df.iloc[:-1, 1:].values
    distances = euclidean_distances(query_vector, other_vectors)
    k=100
    k_indices = np.argsort(distances)[0][:k]
    rec_ids=tfidf_df["id"][k_indices]
    rec_ids=list(rec_ids)
    rec_idss=rec_ids
    movs=movies.loc[movies['id'].isin(rec_ids)]
    id_to_title=dict(zip(movies['id'], movies['title']))

    # input_movie_id=[movie_title_to_id[id] for id in titles]
    # input_movie_id=int(input_movie_id[0])



    
    if lang!="skip":
        # rec_ids=get_recs_lang(lang,movs)
        L = []
        for i, x in enumerate(movs["original_language"]):
            if lang==x:
                L.append(i)
        if not L:
            print("Movie not found lang")
        rec_idss=movs['id'].iloc[L]
        movs=movs.loc[movs['id'].isin(rec_idss)]


    
    if a==0 and combo!="skip":
        # rec_ids=get_recs_genre(combo,movs)
        L = []
        for i, x in enumerate(movs["genres"]):
            if combo in x:
                L.append(i)
        if not L:
            print("Movie not found genre")
        rec_idss=movs['id'].iloc[L]
        movs=movs.loc[movs['id'].isin(rec_idss)]
        

    
    elif a==1 and combo!="skip":
        # rec_ids=get_recs_dir(combo,movs)
        L = []
        for i, x in enumerate(movs["directer"]):
            if combo in x:
                L.append(i)
        if not L:
            print("Movie not found genre")
        rec_idss=movs['id'].iloc[L]
        movs=movs.loc[movs['id'].isin(rec_idss)]

    
    elif a==2 and combo!="skip":
        # rec_ids=get_recs_actor(combo,movs)
        L = []
        for i, x in enumerate(movs["cast"]):
            if combo in x:
                L.append(i)
        if not L:
            print("Movie not found actor")
        rec_idss=movs['id'].iloc[L]
        movs=movs.loc[movs['id'].isin(rec_idss)]
    
    
    # rec_ids=list(rec_ids)
    # rec_idss=pd.Series(rec_idss)
    # arec_ids=np.array(rec_ids)
    # arec_idss=np.array(rec_idss)
    # positions = np.searchsorted(arec_ids,arec_idss)
    # extracted_elements = rec_ids.iloc[positions]
    # print(extracted_elements,rec_ids,rec_idss)
    # rec_idss=sorted(rec_idss,key=lambda x: rec_ids.index(x))

    # movs['temp_sort'] = movs['id'].map(dict(zip(rec_ids, range(len(rec_ids)))))
    # sort_movies = movs.sort_values(by='temp_sort')
    # sort_movies.drop(columns=['temp_sort'], inplace=True)
    # recs=sort_movies["title"][:10]
    # recs = [id_to_title.get(id) for id in rec_idss if id in id_to_title]

    
    rec_idss=list(rec_idss)
    # print(rec_ids,"/n/n",rec_idss)
    rec_idss1 = sorted(rec_idss, key=lambda x: rec_ids.index(x))
    # print(rec_idss1)
    recs = [id_to_title.get(id) for id in rec_idss1 if id in id_to_title]
        
        


    input_movie_titles = titles
    if len(input_movie_titles) == 0:
        print("No movie titles provided. Cannot make recommendations without input.")
    else:
        valid_movie_ids = [movie_title_to_id[title] for title in input_movie_titles if title in movie_title_to_id]

        if len(valid_movie_ids) == 0:
            print("None of the provided movie titles were found in the database. Cannot make recommendations without valid input.")
        else:
            combined_similarities = np.zeros(ratings.shape[0])
            for movie_id in valid_movie_ids:
                if movie_id in ratings.columns:
                    highest_rated_users = ratings[movie_id].idxmax()
                    similarities = cosine_similarity(ratings)
                    combined_similarities += similarities[highest_rated_users]
                # else:
                #     print(f"Warning: Movie ID {movie_id} not found in ratings.")

            most_similar_users = ratings.index[np.argsort(combined_similarities)[::-1]]
            recommended_movies = set()
            for user in most_similar_users:
                unseen_movies = ratings.columns[(ratings.loc[user] == 0) & (~ratings.columns.isin(valid_movie_ids))]
                for movie in unseen_movies:
                    recommended_movies.add(movie)
                    if len(recommended_movies) >= 10:
                        break
            
                # reco_ids = list(recommended_movies)[:10]
                # recs2= [id_to_title.get(id) for id in reco_ids if id in id_to_title]

                recommended_movie_ids = list(recommended_movies)[:10]
                rec_title = [movie_id_to_title[movie_id] for movie_id in recommended_movie_ids if movie_id in movie_id_to_title]
        
        return recs ,rec_title


# In[43]:


L="Harry Potter and the Prisoner of Azkaban,Harry Potter and the Goblet of Fire"
reccs,reccs2=get_recs(L,"English",0,"A dventure",movies,tfidf_df)
reccs=reccs[:11]
reccs2=reccs2[:11]
print(reccs)
print(reccs2)


# In[ ]:


L="John Wick,John Wick: Chapter 2"
reccs,reccs2=get_recs(L,"skip",0,"skip",movies,tfidf_df)
reccs=reccs[:11]
reccs2=reccs2[:11]
print(reccs)
print(reccs2)


# In[40]:


lang="Italian"
lang_to_lag={"English":"en","French":"fr","Italian":"it","Japanese":"ja","German":"de","Spanish":"es","Russian":"ru","Hindi":"hi","Korean":"ko","Chinese":"zh"}
lang=lang_to_lag[lang]
print(lang)


# In[37]:


# ratings=pd.read_csv('ratings_small.csv')
# movie_id_to_title=dict(zip(movies['id'].astype(str), movies['title']))
# movie_title_to_id=dict(zip(movies['title'].astype(str), movies['id']))
# ratings = ratings.drop('timestamp', axis=1)
# ratings['rating'].fillna(0, inplace=True)
# ratings = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
# ratings.columns = ratings.columns.astype(str)


# In[38]:


# # L=["Harry Potter and the Prisoner of Azkaban"]
# input_movie_id=[movie_title_to_id[id] for id in L]
# input_movie_id=int(input_movie_id[0])
# if input_movie_id in ratings.columns:
#     highest_rated_users = ratings[input_movie_id].idxmax()
#     similarities = cosine_similarity(ratings)
#     most_similar_users = ratings.index[np.argsort(similarities[highest_rated_users])[::-1]]
#     recommended_movies = set()

#     for user in most_similar_users:
#         unseen_movies = ratings.columns[(ratings.loc[user] > 0) & (ratings.loc[highest_rated_users] == 0)]
#         for movie in unseen_movies:
#             recommended_movies.add(movie)
#         if len(recommended_movies) >= 10:
#             break   
#     recommended_movie_ids = list(recommended_movies)[:10]
#     recommended_movie_titles = [movie_id_to_title[movie_id] for movie_id in recommended_movie_ids if movie_id in movie_id_to_title]
#     return recommended_movie_titles


# In[39]:


# all_genres = [genre for genres_list in movies['genres'] for genre in genres_list]

# # Get unique genres using a set
# unique_genres = set(all_genres)

# print(unique_genres)


# In[40]:


# recommended_movie_titles=["Fuck You!"]
# ratings = pd.read_csv('ratings_small.csv')
# movies = pd.read_csv('movies_metadata.csv')
# ratings = ratings.drop('timestamp', axis=1)
# ratings['rating'].fillna(0, inplace=True)
# ratings = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
# ratings.columns = ratings.columns.astype(str)

# L=["Iron Man","Captain America: The First Avenger","Iron Man 2"]
# movie_title_to_id = dict(zip(movies['title'].astype(str), movies['id']))
# movie_id_to_title = dict(zip(movies['id'].astype(str), movies['title']))
# id_collab=[movie_title_to_id[id] for id in L]
# input_movie_id=id_collab

# if input_movie_id in ratings.columns:
#     highest_rated_user = ratings[input_movie_id].idxmax()
#     similarities = cosine_similarity(ratings)
#     most_similar_users = ratings.index[np.argsort(similarities[highest_rated_user])[::-1]]
#     recommended_movies = set()

#     for user in most_similar_users:
#         unseen_movies = ratings.columns[(ratings.loc[user] > 0) & (ratings.loc[highest_rated_users] == 0)]
#         for movie in unseen_movies:
#             recommended_movies.add(movie)
#             # if len(recommended_movies) >= 10:
#             #     break
            
#     recommended_movie_ids = list(recommended_movies)#[:10]
#     recommended_movie_titles = [movie_id_to_title[movie_id] for movie_id in recommended_movie_ids if movie_id in movie_id_to_title]

#     print(f"Recommended movies based on user similarity: {recommended_movie_titles}")
# else:
#     print(f"Movie ID  not found in the database.")


# In[41]:


# language_counts = movies['original_language'].value_counts()
# top_10_languages = language_counts.head(10)
# top_10_languages


# In[42]:


# import os
# import ast
# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# ratings = pd.read_csv('ratings_small.csv')
# movies = pd.read_csv('movies_metadata.csv')

# # Create dictionaries mapping movie IDs to titles and vice versa
# movie_id_to_title = dict(zip(movies['id'].astype(str), movies['title']))
# movie_title_to_id = {v: k for k, v in movie_id_to_title.items()}

# ratings = ratings.drop('timestamp', axis=1)
# ratings['rating'].fillna(0, inplace=True)
# ratings = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
# ratings.columns = ratings.columns.astype(str)

# L="Harry Potter and the Prisoner of Azkaban","Harry Potter and the Goblet of Fire"
# input_movie_titles=L

# # Convert titles to their corresponding movie IDs
# valid_movie_ids = [movie_title_to_id[title] for title in input_movie_titles if title in movie_title_to_id]

# if len(valid_movie_ids) == 0:
#     print("None of the provided movie titles were found in the database.")
# else:
#     combined_similarities = np.zeros(ratings.shape[0])
    
#     for movie_id in valid_movie_ids:
#         if movie_id in ratings.columns:
#             highest_rated_users = ratings[movie_id].idxmax()
#             similarities = cosine_similarity(ratings)
#             combined_similarities += similarities[highest_rated_users]
#         else:
#             print(f"Warning: Movie ID {movie_id} not found in ratings.")

#     most_similar_users = ratings.index[np.argsort(combined_similarities)[::-1]]
    
#     recommended_movies = set()

#     for user in most_similar_users:
#         unseen_movies = ratings.columns[(ratings.loc[user] == 0) & (~ratings.columns.isin(valid_movie_ids))]
#         for movie in unseen_movies:
#             recommended_movies.add(movie)

#         if len(recommended_movies) >= 10:
#             break
            
#     recommended_movie_ids = list(recommended_movies)[:10]
#     recommended_movie_titles = [movie_id_to_title[movie_id] for movie_id in recommended_movie_ids if movie_id in movie_id_to_title]
    
#     print(f"Recommended movies based on user similarity: {recommended_movie_titles}")
    


# In[32]:


large_sorted_list = [2, 5, 7, 9, 11]  # Example larger sorted list
small_list = [7, 2, 9]  # Example smaller list to be sorted

# Sort the small list based on the order of the large sorted list
sorted_small_list = sorted(small_list, key=lambda x: large_sorted_list.index(x))

print(sorted_small_list)


# In[ ]:




