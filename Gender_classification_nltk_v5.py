from nltk.corpus import names
import random
import nltk 
from sklearn import cross_validation
# https://www.nltk.org/book/ch06.html

def gender_features2(name):
    features = {}
    features["first_letter"] = name[0].lower()
    features["last_letter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count({})".format(letter)] = name.lower().count(letter)
        features["has({})".format(letter)] = (letter in name.lower())
    return features
#print(gender_features2('Shrek'))

labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
	[(name, 'female') for name in names.words('female.txt')])

random.shuffle(labeled_names)

featuresets = [(gender_features2(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]
#classifier = nltk.NaiveBayesClassifier.train(train_set)
cv = cross_validation.KFold(len(train_set), n_folds=10, shuffle=False, random_state=None)
for traincv, testcv in cv:
	classifier = nltk.NaiveBayesClassifier.train(train_set[traincv[0]:traincv[len(traincv)-1]])
	print ('accuracy:', nltk.classify.util.accuracy(classifier, train_set[testcv[0]:testcv[len(testcv)-1]]))
# Scoring
print(classifier.classify(gender_features2('Neo')))

print(classifier.classify(gender_features2('Trinity')))

# Evaluations
print(nltk.classify.accuracy(classifier, test_set))

# Examine the classifier to determine which features it found most effective;
# likelihood ratios
print(classifier.show_most_informative_features(5))

# use apply_features to use memory economically, when using large corpora;
from nltk.classify import apply_features
train_set = apply_features(gender_features2, labeled_names[500:])
test_set = apply_features(gender_features2, labeled_names[:500])
