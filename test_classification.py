#!/usr/bin/env python
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS 
from sklearn import svm
from gensim.parsing.porter import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
import pandas as pd
import numpy as np
import csv
import timeit

# Read data
train_data = pd.read_csv('train_set.csv', sep='\t', encoding='utf-8')
test_data = pd.read_csv('test_set.csv', sep='\t', encoding='utf-8')

# Subsets
#train_data = train_data[0:100]
#test_data = test_data[0:250]

print 'Starting preprocessing'
start = timeit.default_timer() 

# Stopwords
stopw = set(ENGLISH_STOP_WORDS)

with open('stop_words.txt') as stopfile:
	for word in stopfile:
		stopw.add(word.rstrip())

# Title weight and stemmming title and content for every text in train set
titles = []
sentences = []
i = 0

for title in train_data['Title']:

	title_stem = PorterStemmer().stem_sentence(title)
	titles.append(title_stem)

# print titles

for sentence in train_data['Content']:
	temp_title = ''

	for j in range(10):
		temp_title = titles[i] + ' ' + temp_title

	sentences.append(temp_title + PorterStemmer().stem_sentence(sentence))
	i = i + 1

stop = timeit.default_timer()
print 'Preprocessing time'
print stop - start

# print sentences

le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])

titles2 = []
sentences2 = []
i = 0

# Title weight and stemmming title and content for every text in test set
for title in test_data['Title']:

	title_stem = PorterStemmer().stem_sentence(title)
	titles2.append(title_stem)

# print titles

for sentence in test_data['Content']:
	temp_title = ''

	for j in range(10):
		temp_title = titles2[i] + ' ' + temp_title

	sentences2.append(temp_title + PorterStemmer().stem_sentence(sentence))
	i = i + 1


#Vectorizing-LSI-Classifier
X_train = np.array(sentences)
X_test = np.array(sentences2)
clf = Pipeline([('tfidf', TfidfVectorizer(stop_words=stopw)),\
                ('svd' , TruncatedSVD(n_components=1000) ),\
                ('clf', svm.SVC(C=10, gamma = 0.0001, kernel= 'linear', class_weight='balanced')),
               ])

clf.fit(X_train, y)
predicted = clf.predict(X_test)


#Print Results
categories = le.inverse_transform(predicted)

i = 0
CsvData2 = [['Id', 'Category']]

for t in test_data['Id']:
	CsvData2.append([ t, categories[i]])
	i = i + 1

myCsv2 = open('testSet_categories.csv', 'w')

print 'Writing to csv file \'testSet_categories.csv\' the category results'
with myCsv2:
    writer = csv.writer(myCsv2)
    writer.writerows(CsvData2)
