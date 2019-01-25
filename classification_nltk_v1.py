
import nltk
nltk.usage(nltk.classify.ClassifierI)

train = [
        (dict(a=1,b=1,c=1), 'y'),
        (dict(a=1,b=1,c=1), 'x'),
        (dict(a=1,b=1,c=0), 'y'),
        (dict(a=0,b=1,c=1), 'x'),
        (dict(a=0,b=1,c=1), 'y'),
        (dict(a=0,b=0,c=1), 'y'),
        (dict(a=0,b=1,c=0), 'x'),
        (dict(a=0,b=0,c=0), 'x'),
        (dict(a=0,b=1,c=1), 'y'),
        ]
test = [
        (dict(a=1,b=0,c=1)), # unseen
        (dict(a=1,b=0,c=0)), # unseen
        (dict(a=0,b=1,c=1)), # seen 3 times, labels=y,y,x
        (dict(a=0,b=1,c=0)), # seen 1 time, label=x
        ]
classifier = nltk.classify.NaiveBayesClassifier.train(train)
sorted(classifier.labels())

classifier.classify_many(test)

for pdist in classifier.prob_classify_many(test):
     print('%.4f %.4f' % (pdist.prob('x'), pdist.prob('y')))

classifier.show_most_informative_features()


classifier = nltk.classify.DecisionTreeClassifier.train(
        train, entropy_cutoff=0, support_cutoff=0)
sorted(classifier.labels())

print(classifier)

classifier.classify_many(test)

print(classifier)
'''
for pdist in classifier.prob_classify_many(test):
    print('%.4f %.4f' % (pdist.prob('x'), pdist.prob('y')))'''