import nltk
from nltk.collocations import *

bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

print('Student\'s t: examples from Manning and Schutze 5.3.2:')
print('%0.4f' % bigram_measures.student_t(8, (15828, 4675), 14307668))


print('%0.4f' % bigram_measures.student_t(20, (42, 20), 14307668))

print('Chi-square: examples from Manning and Schutze 5.3.3:')
print('%0.2f' % bigram_measures.chi_sq(8, (15828, 4675), 14307668))


print('%0.0f' % bigram_measures.chi_sq(59, (67, 65), 571007))


print('Likelihood Ratios:')
print('%0.2f' % bigram_measures.likelihood_ratio(110, (2552, 221), 31777))

print('%0.2f' % bigram_measures.likelihood_ratio(8, (13, 32), 31777))

print('Pointwise Mutual Information: examples from Manning and Schutze 5.4:')

print('%0.2f' % bigram_measures.pmi(20, (42, 20), 14307668))
print('%0.2f' % bigram_measures.pmi(20, (15019, 15629), 14307668))
