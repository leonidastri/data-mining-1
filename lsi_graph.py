#!/usr/bin/env python
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score
from sklearn.model_selection import KFold
from sklearn.decomposition import TruncatedSVD
from gensim.parsing.porter import PorterStemmer
from sklearn import preprocessing
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import ourknn as knn
import pandas as pd
import numpy as np
import csv
import timeit

# Read data
train_data = pd.read_csv('train_set.csv', sep='\t')
#test_data = pd.read_csv('test_set.csv', sep='\t')

# Subsets
#train_data = train_data[0:100]
#test_data = test_data[0:250]

print 'Starting preprocessing for LSI'
start = timeit.default_timer() 

# Stopwords
stopw = set(ENGLISH_STOP_WORDS)

with open('stop_words.txt') as stopfile:
	for word in stopfile:
		stopw.add(word.rstrip())

# Title weight and stemmming title and content for every text
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
print 'Ending Preprocessing in'
print stop - start

# print sentences

le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])

tfidf = TfidfVectorizer(stop_words=stopw)
tfidf.fit_transform(np.array(sentences))

classifiers = ['RandomForest', 'SVM', 'KNN']

print "Starting LSI graph"

kf = KFold(n_splits=10)

for c in classifiers:
	start = timeit.default_timer()
	accuracy = []
	components = []
	if c == 'SVM':
		print 'Running SVM'
		svc = svm.SVC(class_weight='balanced')
		parameters = {'C': [1, 10, 100],
    				'gamma': [0.0001, 0.001],
    				'kernel': ['linear','rbf'] }
		clf = GridSearchCV(svc, parameters,n_jobs=-1)
	elif c == 'RandomForest':
		print 'Running Random Forest'
		clf = RandomForestClassifier(n_jobs=-1)
	elif c == 'KNN':
		print 'Running KNN'

	for i in range(7):
		if i == 0:
			components.append((i+1) * 100)
		else:
			components.append((2*i) * 100)

		svd = TruncatedSVD(n_components=components[i])
		acc=0
		for train_index, test_index in kf.split(np.array(sentences)):
			X_train = tfidf.transform(np.array(sentences)[train_index])
			X_test = tfidf.transform(np.array(sentences)[test_index])

			#SVD transform
			X_train = svd.fit_transform(X_train)
			X_test = svd.transform(X_test)

			if c != 'KNN' :
				clf.fit(X_train, y[train_index])
				ypred = clf.predict(X_test)
			else:
				ypred = knn.myKnn(X_test, X_train, y[train_index], 5)

			acc += accuracy_score(y[test_index], ypred)
		accuracy.append(acc/10)
	plt.title('LSI Accuracy for ' + c + ' classifier')
	plt.plot(components, accuracy, linewidth = 1.5, linestyle = '-')
	plt.ylabel('Accuracy')
	plt.xlabel('Components')
	plt.savefig(c + '.png')
	plt.close()
	#plt.show()
			
	stop = timeit.default_timer()
	print 'Ending LSI for ' + c + ' classifier with time ' + str(stop-start)
