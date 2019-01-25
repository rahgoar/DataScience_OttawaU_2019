from nltk.corpus import names
import random
import nltk 

# https://www.nltk.org/book/ch06.html

def gender_features(word):
	return {'last_letter': word[-1]}
print(gender_features('Shrek'))

labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
	[(name, 'female') for name in names.words('female.txt')])

random.shuffle(labeled_names)

featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Scoring
print(classifier.classify(gender_features('Neo')))

print(classifier.classify(gender_features('Trinity')))

# Evaluations
print(nltk.classify.accuracy(classifier, test_set))

# Examine the classifier to determine which features it found most effective;
# likelihood ratios
print(classifier.show_most_informative_features(5))

# use apply_features to use memory economically, when using large corpora;
from nltk.classify import apply_features
train_set = apply_features(gender_features, labeled_names[500:])
test_set = apply_features(gender_features, labeled_names[:500])
