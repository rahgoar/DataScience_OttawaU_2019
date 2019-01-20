import nltk
from nltk.collocations import *

bigram_measures = nltk.collocations.BigramAssocMeasures()

trigram_measures = nltk.collocations.TrigramAssocMeasures()
finder = BigramCollocationFinder.from_words(
	nltk.corpus.genesis.words('english-web.txt'))

#top ten bigram collocations in Genesis
print(finder.nbest(bigram_measures.pmi, 10))  # doctest: +NORMALIZE_WHITESPACE

print('------------Should Occure > 3 times----------')
#ignoring all bigrams which occur less than three times;
finder.apply_freq_filter(3)
print(finder.nbest(bigram_measures.pmi, 10))

print('---------Only Tags-------------')

finder = BigramCollocationFinder.from_words(t for w, t in
	nltk.corpus.brown.tagged_words('ca01', tagset='universal'))
print(finder.nbest(bigram_measures.pmi, 10) ) # doctest: +NORMALIZE_WHITESPACE
print('----------------------')

