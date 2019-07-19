import time


from sklearn.datasets import load_svmlight_file
from sklearn.naive_bayes import MultinomialNB #GaussianNB

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from numpy import shape
from sklearn.model_selection import cross_val_score

from code.statistics import statistics
from code.classifiers import EncodedClassifier


train_file = "C:/Users/Cristina/PhD Track/Data/LSHTC/LSHTC1/dry-run/train_corrected.txt"
test_file = "C:/Users/Cristina/PhD Track/Data/LSHTC/LSHTC1/dry-run/test_corrected.txt"

X_train, y_train = load_svmlight_file(train_file)
X_test, y_test = load_svmlight_file(test_file)


if shape(X_train)[1] > shape(X_test)[1]:
    X_test, y_test = load_svmlight_file(test_file, n_features=shape(X_train)[1])
else:
    X_train, y_train = load_svmlight_file(train_file, n_features=shape(X_test)[1])

y_train = [str(int(i)) for i in y_train]
y_test = [str(int(i)) for i in y_test]


for encoding_type in ["magic-tree"]:
    print(encoding_type)
    cls = EncodedClassifier(SVC(class_weight="balanced", random_state=1, ), encoding_type=encoding_type)
    cls.fit(X_train, y_train)
    y_pred = cls.predict(X_test)
    statistics(y_train, y_pred, y_test, show_hist=False)
