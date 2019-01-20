
from nltk.corpus import gutenberg   # Docs from project gutenberg.org

files_en = gutenberg.fileids()      # Get file ids
doc_en = gutenberg.open('austen-emma.txt').read()


from nltk import regexp_tokenize
pattern = r'''(?x) ([A-Z]\.)+ | \w+(-\w+)* | \$?\d+(\.\d+)?%? | \.\.\. | [][.,;"'?():-_`]'''
tokens_en = regexp_tokenize(doc_en, pattern)
#nltk.download('gutenberg')

import nltk
en = nltk.Text(tokens_en)

print(len(en.tokens))       # returns number of tokens (document length)
print(len(set(en.tokens)))  # returns number of unique tokens
en.vocab() 

#en.plot(50)
print(doc_en.count('Emma'))
print(tokens_en.count('Emma'))
print(en.count('Emma'))        # Counts occurrences

#en.dispersion_plot(['Emma', 'Frank', 'Jane'])

#en.concordance('Emma', lines=5)

# Find similar words;
#en.similar('Emma')
#en.similar('Frank')

#en.collocations()