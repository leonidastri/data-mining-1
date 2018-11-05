#!/usr/bin/env python
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
from sklearn.model_selection import KFold
from gensim.parsing.porter import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
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

print 'Starting preprocessing'
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
print 'Preprocessing time'
print stop - start

# print sentences

le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])

tfidf = TfidfVectorizer(stop_words=stopw)
tfidf.fit_transform(np.array(sentences))

classifiers = [ 'MultinomialNaiveBayes', 'RandomForest', 'SVM', 'KNN', 'SVC' ]
ac_list, ps_list, rs_list, fs_list = ['Accuracy'], ['Precision'], ['Recall'], ['F-Measure']

start = timeit.default_timer() 

for c in classifiers: 
	if c == 'SVM':
		print 'Running SVM'
		svc = svm.SVC(class_weight='balanced')
		parameters = {'C': [1, 10, 100],
        			'gamma': [0.0001, 0.001],
        			'kernel': ['linear','rbf'] }
		#clf = GridSearchCV(svc, parameters,n_jobs=-1)
		clf = Pipeline([('tfidf', TfidfVectorizer(stop_words=stopw)),\
                ('svd' , TruncatedSVD(n_components=200) ),\
                ('clf', GridSearchCV(svc, parameters, n_jobs=-1)),
                ])
	elif c == 'RandomForest':
		print 'Running Random Forest'
		#clf = RandomForestClassifier(n_jobs=-1 )
		clf = Pipeline([('tfidf', TfidfVectorizer(stop_words=stopw)),\
                ('svd' , TruncatedSVD(n_components=50) ),\
                ('clf', RandomForestClassifier(n_jobs=-1)),
                ])
	elif c == 'MultinomialNaiveBayes':
		print 'Running Multinomial Naive Bayes'
		#clf = MultinomialNB()
		#MNB doesn't have SVD
		clf = Pipeline([('tfidf', TfidfVectorizer(stop_words=stopw)),\
                ('clf', MultinomialNB()),
               ])
	elif c == 'KNN':
		print 'Running KNN'
	elif c == 'SVC':
		print 'Running SVC'
		clf = Pipeline([('tfidf', TfidfVectorizer(stop_words=stopw)),\
                   ('svd' , TruncatedSVD(n_components=1000) ),\
                   ('clf', svm.SVC(C=10, gamma = 0.0001, kernel= 'linear', class_weight='balanced')),
                   ])

	kf = KFold(n_splits=10)
	ac, ps, rs, fs = 0, 0, 0, 0

	for train_index, test_index in kf.split(np.array(sentences)):
		X_train = np.array(sentences)[train_index]
		X_test = np.array(sentences)[test_index]

		if c != 'KNN' :
			clf.fit(X_train, y[train_index])
			predicted = clf.predict(X_test)
		else:
			svd = TruncatedSVD(n_components=50)
			X_train = svd.fit_transform(tfidf.transform(np.array(sentences)[train_index]))
			X_test = svd.transform(tfidf.transform(np.array(sentences)[test_index]))
			predicted = knn.myKnn(X_test, X_train, y[train_index], 5)

		ac += accuracy_score(y[test_index], predicted)
		ps += precision_score(y[test_index], predicted, average='macro')
		rs += recall_score(y[test_index], predicted, average='macro')
		fs += f1_score(y[test_index], predicted, average='macro')
		print(classification_report(predicted, y[test_index], target_names=list(le.classes_)))
	ac_list.append(ac/10)
	ps_list.append(ps/10)
	rs_list.append(rs/10)
	fs_list.append(fs/10)

stop = timeit.default_timer()
print 'Time'
print stop - start

CsvData = [['Statistic Measure', 'Naive Bayes', 'Random Forest', 'SVM', 'KNN', 'SVC'], ac_list, ps_list, rs_list, fs_list]
 
myCsv= open('EvaluationMetric_10fold.csv', 'w')

print 'Writing to csv file \'EvaluationMetric_10fold.csv\' the metric results'
with myCsv:
    writer = csv.writer(myCsv)
    writer.writerows(CsvData)
