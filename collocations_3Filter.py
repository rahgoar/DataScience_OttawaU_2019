import nltk
from nltk.collocations import *

bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

text = "I do not like green eggs and ham, I do not like them Sam I am!"
tokens = nltk.wordpunct_tokenize(text)
print('----Perhaps all n-grams are too many to find collocations-----')
finder = TrigramCollocationFinder.from_words(tokens)
print(len(finder.score_ngrams(trigram_measures.raw_freq)))

print('----let\'s remove personal pronouns -----')
finder.apply_word_filter(lambda w: w in ('I', 'me'))
print(len(finder.score_ngrams(trigram_measures.raw_freq)))

print('-----Let\'s see the candidate which occurs more than once:')
print(sorted(finder.above_score(trigram_measures.raw_freq,
	1.0 / len(tuple(nltk.trigrams(tokens))))))

print('----------permit \'and\' to appear in the middle of a trigram, but not on either edge--------')

finder.apply_ngram_filter(lambda w1, w2, w3: 'and' in (w1, w3))
print(len(finder.score_ngrams(trigram_measures.raw_freq)))

print('------Remove low frequency items, (probably less important)-----------')
finder.apply_freq_filter(2)
print(len(finder.score_ngrams(trigram_measures.raw_freq)))