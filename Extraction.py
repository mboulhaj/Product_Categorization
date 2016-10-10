#!/usr/bin/env python
# -*- coding :utf-8 -*-

import requests
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.naive_bayes import GaussianNB
import pickle
import operator
import sklearn
from sklearn import tree


# dict Vectorizer
vec = DictVectorizer()

# cont Vectorizer
cont_vect1 = CountVectorizer(min_df=1) # name
cont_vect2 = CountVectorizer(min_df=1) # descrption

# transformer
transformer = TfidfTransformer(smooth_idf=False)

# Naive Bayes
gnb = GaussianNB()



url ='http://kaioshin.fastupcommerce.com/api/v1'
token='Token token=f7eb0628b18c00a81ab098152bb676eb'
nb_pages=655
nb_outputfiles=20
model =''


# get product from kaioshin
def get_products(url, token):
    i=0    # page
    while i < nb_pages+1:

        # one page
        products = requests.get(url+'/products??country=uk&marketplace=amazon&page='+'/'+str(i), headers={'Authorization': token})
        for product in products.json()['products']:

            # product attributes [id, description, category_name]
            vector = [pre_process_text(product['name']), pre_process_text(product['description'])]

            # highest categorization level
            Class=product['category']['parents'][-1]
            yield vector, Class
        i+=1


# preprocess textual fields
def pre_process_text(text):

    # Tokenazition and Lowercase
    # Stopword, symbols and number removal

    tokens_description = word_tokenize(text)
    tokens_description=[re.sub(r'[^\w]', '', s) for s in tokens_description]    #Symbol removal
    tokens_description=[ re.sub("\d+", " ", s) for s in tokens_description]    #Number removal

    # Blank space removal
    while (' ' in tokens_description):
        i=tokens_description.index(' ')
        del tokens_description[i]

    # Stemming
    stemmer = SnowballStemmer("english")
    stem_description=[stemmer.stem(a) for a in tokens_description if a.lower() not in stopwords.words('english') ]


    return ' '.join(stem_description)


# vectorize
def vectorize(array, data):

    # cont vectorize
    if data =="name":
        cont_array = cont_vect1.fit_transform(array).toarray()
        pickle.dump(cont_vect1.vocabulary_,open("feature1.pkl","wb"))
    if data =="description":
        cont_array = cont_vect2.fit_transform(array).toarray()
        pickle.dump(cont_vect2.vocabulary_,open("feature2.pkl","wb"))
    # tfidf transform
    tfidf_array = transformer.fit_transform(cont_array).toarray()

    return tfidf_array

# vectorize test
# forcer le vocabulaire (a revoir)
def vectorize_test(array, data):

    if data =="name":
        loaded_vec = CountVectorizer(vocabulary=pickle.load(open("feature1.pkl", "rb")))
    if data =="description":
        loaded_vec = CountVectorizer(vocabulary=pickle.load(open("feature2.pkl", "rb")))
    tfidf = transformer.fit_transform(loaded_vec.fit_transform(array)).toarray()
    return tfidf




def build_model():
    name_array=[]
    description_array=[]

    # extract and preprocess
    Class=[]
    for vector, c in get_products(url,token):

        name_array.append(vector[0])
        description_array.append(vector[1])
        Class.append(c)

    # vectorize
    names=vectorize(name_array, "name")
    descriptions=vectorize(description_array,"description")

    # build training vectors
    k=0  # 0 if first vector
    a=0  # inctement for output files

    filenames=[]    # array containing output files

    while(a<nb_outputfiles):
        filenames.append(open('vectors'+str(a),'w'))
        for feature in sorted(cont_vect1.vocabulary_.items(), key=operator.itemgetter(1)):
            filenames[a].write(str(feature[0])+',')
        for feature in sorted(cont_vect2.vocabulary_.items(), key=operator.itemgetter(1)):
            filenames[a].write(str(feature[0])+',')
        filenames[a].write('\n')

        a+=1

    inc=0       # if inc== 1000, change file
    current=0

    for name,description in zip(names, descriptions):

        if inc == 1000:
            inc=0
            current+=1

        # increment product nb
        inc+=1

        vector=name.tolist()
        vector.extend(description.tolist())

        # write in file
        for element in vector :
            filenames[current].write( str(element) +'\t')
        filenames[current].write('\n')


        if k==0:
            vectors= np.array(np.array(vector))
            k=1
        else :
            vectors=np.vstack((vectors, np.array(vector)))


    # Build ML model
    #clf = tree.DecisionTreeClassifier()
    #model = clf.fit(vectors, Class)


    # Build Naive Bayes Model
    model = gnb.fit(vectors, Class)

    return model











