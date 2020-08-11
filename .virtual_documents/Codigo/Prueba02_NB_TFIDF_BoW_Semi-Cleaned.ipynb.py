import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV


#Cargamos dataset train en un dataframe
tweets_train = pd.read_csv('Dataset/train.csv')
tweets_test = pd.read_csv('Dataset/test.csv')
tweets_submission = pd.read_csv('Dataset/sample_submission.csv') 


tweets_train.head(10)


#Eliminamos los emojis del texto
def remove_emoji(text):
    emoji_list = re.compile("["u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\u200d"
                               u"\u2640-\u2642"
                           "]+", flags=re.UNICODE)
    return emoji_list.sub(r'', text)


#Almacenamos el texto limpio en una nueva feature del dataframe
tweets_train['cleaned_text']=tweets_train['text'].apply(lambda x: remove_emoji(x))


#Eliminamos los numeros y '_' del texto 
tweets_train['cleaned_text']=tweets_train['cleaned_text'].apply(lambda x: re.sub(r'[0-9_]','',x))
tweets_test['cleaned_text']=tweets_test['text'].apply(lambda x: re.sub(r'[0-9_]','',x))


tweets_train.head(10)


tweets_test.head(10)


x = tweets_train.cleaned_text
y = tweets_train.target


#Separamos al dataset tweets_train en un set de entrenamiento y uno de validacion, para text y para target
#El tamaño del set de validacion es del 20% del original
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


#Bag of Words 
#Vectorizamos los textos de cada tweet
count_vect = CountVectorizer(stop_words = ('english'), lowercase = True)
tweets_train_counts = count_vect.fit_transform(x_train)
tweets_test_counts = count_vect.transform(x_test)
tweets_train_counts.shape
#print(count_vect.get_feature_names())


#TFIDF
tfidf_transformer = TfidfTransformer()
tweets_train_tfidf = tfidf_transformer.fit_transform(tweets_train_counts)
tweets_test_tfidf = tfidf_transformer.transform(tweets_test_counts)
tweets_train_tfidf.shape


#Naive Bayes
tweets_train_NB = MultinomialNB().fit(tweets_train_tfidf, y_train)


predicted = tweets_train_NB.predict(tweets_test_tfidf)
np.mean(predicted == y_test)


#Realizamos gridsearch para buscar el mejor hiperparametro
parameters = {'alpha': ([0.001,0.1,0.3,0.5,0.6,1])}

gs_NB = GridSearchCV(tweets_train_NB, parameters, n_jobs=-1, cv=3)
gs_NB = gs_NB.fit(tweets_train_tfidf, y_train)

gs_NB_best = gs_NB.best_estimator_
gs_NB.best_score_
gs_NB.best_params_


best_NB = gs_NB_best.fit(tweets_train_tfidf, y_train)


predicted = best_NB.predict(tweets_test_tfidf)
np.mean(predicted == y_test)


test_text = tweets_test.cleaned_text
test_text_counts = count_vect.transform(test_text)
test_text_tfidf = tfidf_transformer.transform(test_text_counts)
test_target_predicted = best_NB.predict(test_text_tfidf)


tweets_submission.target = test_target_predicted
tweets_submission.to_csv("submission.csv",index=False)


plt.figure(figsize=(10,7))
g = sns.barplot(x= tweets_submission.target.value_counts().index, y= tweets_submission.target.value_counts().values, orient='v', palette= 'husl', hue= tweets_submission.target.value_counts().index, dodge=False)
g.set_title("Tweets Test", fontsize=22)
g.set_xlabel("Tipo de noticia", fontsize=16)
g.set_ylabel("Cantidad de tweets", fontsize=16)



