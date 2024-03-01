MOVIE RECOMENDDATION SYSTEM USING ML ALGORITHMS


Dataset used : The Movies Dataset | Kaggle 

Data Cleaning :
As Data cleaning is used to improve quality and reliability of the dataset.We handled 
following errors : 
1. Missing values: This may occur due to various reasons like entry errors,incomplete 
records,etc.We dropped such rows and columns containing NULL values. 
2.Converting Data types: Ensuring that variables have appropriate data types is important for 
analysis and modeling.eg. In this model we converted the data type of movie id to string. 
3.Standardizing Data to the same format: Removing spaces between words,converting all text to 
lowercase,etc. comes under this category.This brought uniformity and made comparison task 
easy. 


Content-Based Filtering

 Extraction of Data :

 We extracted data such as actors, directors, keywords from different csv files.We 
converted directors’ and overviews’ data type to string.Then we merged genres,cast,overview 
and keywords under one column named ‘tags’ for efficient handling of the data. 


Applying TF-IDF Vectoriser :

We applied tfidf vectoriser on column named ‘tags’ in content-based filtering. 


Centroid and Euclidean Distances :


In a multi dimensional vector space by finding the centroid of the input movies we apply a 
combined effect based on users preferences. Euclidean distances are often used as a similarity 
measure to compare the distances between data points.So, here we applied it to find the distances 
between movies and the centroid. 


K-Nearest Neighbors (KNN) Technique :


After finding the distances and selecting a value for k (usually large to prevent noise) , we 
applied the KNN algorithm to find the nearest movies

![image](https://github.com/KULTHEOM/Movie-recommendation-system-/assets/142748736/68686e03-85b1-439e-9ffa-2843ace7ff69)


Collaborative Filtering:

 Extraction of Data :


 We extracted data of ratings and movies from csv files.Then we converted movie id to titles 
using dictionary mapping and changed the data type of ratings to string.


![image](https://github.com/KULTHEOM/Movie-recommendation-system-/assets/142748736/c6bafc15-3ab5-4068-af49-2bbee1620b1a)



Applying cosine similarity :

We created a pivot table with user id, movie id and ratings to find distribution of different 
movie ratings by users.Cosine similarity is applied on ratings.We found the similarity between 
highest rated users and on comparing we enlisted the unseen movies to be recommended to them. 

![image](https://github.com/KULTHEOM/Movie-recommendation-system-/assets/142748736/e1b48f83-9eb6-416a-8032-45303b42c335)



Recommendation

Inputs coming through chat bot is taken and accordingly different recommendation functions are 
called and output is displayed. 

                                       TO MAKE CHATBOT FOR IMPLEMENTATION 

Used the telegram of Bot father and written code of it . 
                                       



 


        
