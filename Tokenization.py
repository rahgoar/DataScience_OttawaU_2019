# -*- coding: UTF-8 -*-
import nltk
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

text = nltk.corpus.gutenberg.raw('chesterton-thursday.txt')
tokenized_sents=nltk.sent_tokenize(text)
#print(tokenized_sents)
tokenized_word=nltk.word_tokenize(text)
fdist = FreqDist(tokenized_word)
print(fdist)
print(fdist.most_common(2))

fdist.plot(30,cumulative=False)
plt.show()


stop_words=set(stopwords.words("english"))
print(stop_words)


filtered_sents=[]
for w in tokenized_sents:
    if w not in stop_words:
        filtered_sents.append(w)
#print("Tokenized Sentence:",tokenized_sents)
#print("Filterd Sentence:",filtered_sents)

fdist = FreqDist(filtered_sents)
print(fdist)
print(fdist.most_common(2))

fdist.plot(30,cumulative=False)
plt.show()

