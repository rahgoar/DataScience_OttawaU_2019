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



labeled_docs = [(doc, 'chesterton-thursday') for doc in tokenized_sents]

print(labeled_docs)
