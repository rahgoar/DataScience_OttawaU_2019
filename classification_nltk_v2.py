from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC

train_data = [({"a": 4, "b": 1, "c": 0}, "ham"),
                  ({"a": 5, "b": 2, "c": 1}, "ham"),
                  ({"a": 0, "b": 3, "c": 4}, "spam"),
                  ({"a": 5, "b": 1, "c": 1}, "ham"),
                  ({"a": 1, "b": 4, "c": 3}, "spam")]

classif = SklearnClassifier(BernoulliNB()).train(train_data)
test_data = [{"a": 3, "b": 2, "c": 1},
             {"a": 0, "b": 3, "c": 7}]

classif.classify_many(test_data)
#print(classif.classify(gender_features('Frank')))
#classif.show_most_informative_features(5)
print(classif.labels())

classif = SklearnClassifier(SVC(), sparse=False).train(train_data)
classif.classify_many(test_data)

print(classif.labels())