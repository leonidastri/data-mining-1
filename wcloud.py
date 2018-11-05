#!/usr/bin/env python
from os import path
from wordcloud import WordCloud , STOPWORDS
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn import preprocessing
import pandas as pd

#Read Data
train_data = pd.read_csv('train_set.csv', sep="\t")

#Subset Data
#train_data = train_data[0:100]

le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])

categories = le.classes_


#Add stop words
stop_words = set(ENGLISH_STOP_WORDS)

with open('stop_words.txt') as stopfile:
	for word in stopfile:
		stop_words.add(word.rstrip())

answer = raw_input("Do you want to add the title to wordcloud?[y/n]")

for cat in categories:
	data = train_data[train_data['Category'] == cat]

	str1 = ''.join(data['Content'])

	if (answer == 'y'):
		str2 = ''.join(data['Title'])
		str3 = str1 + str2
	else:
		str3 = str1

	wordcloud = WordCloud(stopwords=stop_words).generate(str3)
	image = wordcloud.to_file( cat + '.png')
