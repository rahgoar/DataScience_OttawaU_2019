import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer

text = nltk.corpus.gutenberg.raw('chesterton-thursday.txt')
tokenized_sents=nltk.sent_tokenize(text)
#print(tokenized_sents)
tokenized_word=nltk.word_tokenize(text)

stop_words=set(stopwords.words("english"))
print(stop_words)

filtered_sents=[]
for w in tokenized_sents:
    if w not in stop_words:
        filtered_sents.append(w)

ps = PorterStemmer()

stemmed_words=[]
for w in filtered_sents:
    stemmed_words.append(ps.stem(w))

#BOW of stemmed words
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
filtered_counts= cv.fit_transform(filtered_sents)
stemmed_counts= cv.fit_transform(stemmed_words)

word = "flying"
lem = WordNetLemmatizer()
#Examples;
print("Lemmatized Word:",lem.lemmatize(word,"v"))
print("Stemmed Word:",ps.stem(word))

lemmatized_words=[]
for w in filtered_sents:
    lemmatized_words.append(lem.lemmatize(w))
#BOW by lemmatized words;
text_counts2=cv.fit_transform(lemmatized_words)

#producing TFiDF transformation of stemmed and lemmatized words;
tf=TfidfVectorizer()
text_tf= tf.fit_transform(stemmed_words)
text_tf2= tf.fit_transform(lemmatized_words)
