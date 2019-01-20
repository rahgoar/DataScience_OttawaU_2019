import nltk
from nltk.collocations import *

bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

text = "I do not like green eggs and ham, I do not like them Sam I am!"
tokens = nltk.wordpunct_tokenize(text)
'''
print('------Let\'s see the bigrams:-------')
finder = BigramCollocationFinder.from_words(tokens)
scored = finder.score_ngrams(bigram_measures.raw_freq)
print(sorted(bigram for bigram, score in scored) ) # doctest: +NORMALIZE_WHITESPACE
'''
print('------Let\'s see the trigrams:-------')
finder = TrigramCollocationFinder.from_words(tokens)
scored = finder.score_ngrams(trigram_measures.raw_freq)
#print(scored)
print(sorted(trigram for trigram, score in scored) )

print('------we may want to select only the top n------------')

print(sorted(finder.nbest(trigram_measures.raw_freq, 2)))


print('------we may also want to select only the top score------------')
print(sorted(finder.above_score(trigram_measures.raw_freq,
    1.0 / len(tuple(nltk.trigrams(tokens))))))


print('------Closer look at the n-gram frequencies:------------')
print(sorted(finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))[:10])  # doctest: +NORMALIZE_WHITESPACE

print('----------------------')