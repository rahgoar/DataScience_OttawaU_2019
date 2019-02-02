

# -*- coding: UTF-8 -*-
import nltk

from collections import Counter

text = nltk.corpus.gutenberg.raw('chesterton-thursday.txt')
tokenized_sents=nltk.sent_tokenize(text)
#print(tokenized_sents)
tokenized_word=nltk.word_tokenize(text)
print(len(tokenized_sents))
print(len(tokenized_word))
print(len(tokenized_sents)/len(tokenized_word))

doc = ""
labeled_docs=[]
for i, sent in enumerate(tokenized_sents):
	doc += sent
	if i%5==0:
		labeled_docs.append( [(doc, 'chesterton-thursday') ])
		doc = ""
		
print(labeled_docs)

