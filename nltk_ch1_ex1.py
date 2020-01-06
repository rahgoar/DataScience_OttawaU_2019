import nltk
nltk.download('gutenberg')
nltk.download('punkt')
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

text = nltk.corpus.gutenberg.raw('chesterton-thursday.txt')
tokenized_sents=nltk.sent_tokenize(text)

tokenized_word=nltk.word_tokenize(text)
fdist = FreqDist(tokenized_word)
print(fdist)
print(fdist.most_common(2))

fdist.plot(30,cumulative=False)
plt.show()

#google: what objects are nltk.book import *
#https://stackoverflow.com/questions/17734534/how-do-i-use-the-book-functions-e-g-concoordance-in-nltk

vocab = set(text)
vocab_size = len(vocab)
vocab_size


len(text)

len(set(text))
sorted(set(text))

len(set(text)) / len(text) # measure of lexical richness
#sorted(set(tokenized_word))

####################################################################
#				Clean Data
####################################################################

#nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
filtered_sentence = [w for w in tokenized_word if not w in stop_words]
len(tokenized_word)
#print(word_tokens) 
len(filtered_sentence)
#print(filtered_sentence) 


len(set(tokenized_word)) / len(tokenized_word)

len(set(filtered_sentence)) / len(filtered_sentence)
