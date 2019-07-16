import random
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from scipy import sparse


from code.encoding_decoding import create_balanced_tree_encoding, create_encoding_huffman, \
    create_encoding_random, decode_users, encode_users, encode_classes, Node


# Node with classifier and classes per Node
class ClassificationNode(Node):
    def __init__(self, classes, classifier=None, left_node=None, right_node=None, level=0):
        Node.__init__(self, left_node, right_node, level)
        self.classes = classes
        self.classifier = classifier


def train_classifier(x_train, y_train, x_test, y_test, estimator, max_iterations=10, target_acc=0.8):
    """
    Calculates the best repartition and the corresponding classifier given the data
    :param x_train: the data to be trained on
    :param y_train: the classes for the training data
    :param x_test: the data to be tested on
    :param y_test: the classes for the training data
    :param estimator: the estimator to use as a model
    :param max_iterations: maximum of iterations to do to improve the result
    :param target_acc: accuracy value that is enough to consider the results good enough
    :return: best_est - the estimator corresponding to the best repartition found
             best_sets[0] - first set of the class repartition
             best_sets[1] - second set of the class repartition
    """
    y_train_set = set(y_train)

    # Put in the y_1 randomly half of the elements from the y_train set
    y_1 = set(random.sample(y_train_set, len(y_train_set) // 2))
    y_2 = y_train_set - y_1

    # Create lists for y_train and y_test where classes are 0 or 1 given their inclusion in y_1 or y_2
    y_train_1 = [1 if y in y_1 else 0 for y in y_train]
    y_test_1 = [1 if y in y_1 else 0 for y in y_test]

    # Create a first iteration
    # to initialize the classifier, the maximum accuracy, best repartition of y_1 and y_2 and the best estimator
    current_estimator = clone(estimator)
    current_estimator.fit(x_train, y_train_1)
    predicted = current_estimator.predict(x_test)
    acc = accuracy_score(predicted, y_test_1)
    max_acc = acc
    best_sets = [y_1, y_2]
    best_est = clone(current_estimator)

    count = 1
    # Loop that tries to improve the repartition and classify again in order to improve the results
    while acc < target_acc and count < max_iterations:
        count += 1
        # class_predicted is a dict that keeps the count for every real class how many times 0 and 1 are predicted
        class_predicted = {}
        for j, i in enumerate(y_test):
            if i in class_predicted:
                class_predicted[i][predicted[j]] += 1
            else:
                class_predicted[i] = {}
                class_predicted[i][predicted[j]] = 1
                class_predicted[i][int(not predicted[j])] = 0
        # redo the repartition of classes given the majority prediction
        y_1 = set()
        y_2 = set()
        for i in class_predicted:
            if class_predicted[i][1] > class_predicted[i][0]:
                y_1.add(i)
            else:
                y_2.add(i)

        # if one of the created sets is empty, give it an element from the other set
        if not y_1:
            y_1 = {y_2.pop()}
        elif not y_2:
            y_2 = {y_1.pop()}

        # redo the prediction
        y_train_1 = [1 if y in y_1 else 0 for y in y_train]
        y_test_1 = [1 if y in y_1 else 0 for y in y_test]
        current_estimator.fit(x_train, y_train_1)
        predicted = current_estimator.predict(x_test)
        acc = accuracy_score(predicted, y_test_1)
        if acc > max_acc:
            max_acc = acc
            best_sets = [y_1, y_2]
            best_est = clone(current_estimator)
    #    if max_acc < 0.8:
    #        print("The best acc that could be obtained in 10 iterations is %2.2f " %max_acc)
    return best_est, best_sets[0], best_sets[1]


def node_train(node, x_train, y_train, estimator):
    if sparse.issparse(x_train):
        node_train_sparse(node, x_train, y_train, estimator)
    else:
        node_train_dataframe(node, x_train, y_train, estimator)


def node_train_dataframe(node, x_train, y_train, estimator):
    row_dict_train = {}
    for i, c in enumerate(y_train):
        if c in row_dict_train:
            row_dict_train[c].append(i)
        else:
            row_dict_train[c] = [i]
    if len(node.classes) > 1:
        classifier, y_1, y_2 = train_classifier(x_train, y_train, x_train, y_train, estimator)
        node.classifier = classifier
        y_1_train = [y for y in y_train if y in y_1]
        y_2_train = [y for y in y_train if y in y_2]
        x_1_train = x_train.iloc[sum([row_dict_train[i] for i in y_1], [])]
        x_2_train = x_train.iloc[sum([row_dict_train[i] for i in y_1], [])]
        node.left_node = ClassificationNode(y_1)
        node_train_sparse(node.left_node, x_1_train, y_1_train, estimator)
        node.right_node = ClassificationNode(y_2)
        node_train_sparse(node.right_node, x_2_train, y_2_train, estimator)
        node.change_level(max(node.right_node.level, node.left_node.level) + 1)
    else:
        node.change_level(0)


def node_train_sparse(node, x_train, y_train, estimator):
    row_dict_train = {}
    for i, c in enumerate(y_train):
        if c in row_dict_train:
            row_dict_train[c].append(x_train[i])
        else:
            row_dict_train[c] = [x_train[i]]

    if len(node.classes) > 1:
        classifier, y_1, y_2 = train_classifier(x_train, y_train, x_train, y_train, estimator)
        node.classifier = classifier
        y_1_train = [y for y in y_train if y in y_1]
        y_2_train = [y for y in y_train if y in y_2]
        x_1_train = sparse.vstack([row_dict_train[i].pop(0) for i in y_1_train])
        x_2_train = sparse.vstack([row_dict_train[i].pop(0) for i in y_2_train])
        node.left_node = ClassificationNode(y_1)
        node_train_sparse(node.left_node, x_1_train, y_1_train, estimator)
        node.right_node = ClassificationNode(y_2)
        node_train_sparse(node.right_node, x_2_train, y_2_train, estimator)
        node.change_level(max(node.right_node.level, node.left_node.level) + 1)
    else:
        node.change_level(0)


def create_magic_tree_encoding(x_train, y_train, estimator):
    root = ClassificationNode(set(y_train))
    node_train(root, x_train, y_train, estimator)
    class_to_codes, codes_to_class = encode_classes(root, "", root.level, "classes", True)
    return class_to_codes, codes_to_class, root.level


def fit_single(estimator, x, y):
    estimator = clone(estimator)
    estimator.fit(x, y)
    return estimator


class EncodedClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, estimator, n_jobs=1, encoding_type="random"):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.encoding_type = encoding_type
        self.estimators_ = None

    # y is has Series type
    def __encode(self, X, y):
        if self.encoding_type == "huffman":
            y = pd.Series(y)
            self.dict_user_code, self.dict_code_user, self.encoding_length = create_encoding_huffman(y)
        if self.encoding_type == "balanced-tree":
            y = pd.Series(y)
            self.dict_user_code, self.dict_code_user, self.encoding_length = create_balanced_tree_encoding(y)
        if self.encoding_type == "random":
            self.dict_user_code, self.dict_code_user, self.encoding_length = create_encoding_random(set(y))
        if self.encoding_type == "magic-tree":
            self.dict_user_code, self.dict_code_user, self.encoding_length = create_magic_tree_encoding(X, y,
                                                                                                        self.estimator)
        encoded_train = encode_users(y, self.dict_user_code, self.encoding_length)
        return encoded_train

    def fit(self, x, y=None):
        y = self.__encode(x, y)
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(fit_single)(self.estimator, x, y[column]) for _, column in enumerate(y))
        return self

    def predict(self, x):
        results = np.array([estimator.predict(x) for estimator in self.estimators_]).T
        results = pd.DataFrame(results)
        y_pred = decode_users(results, self.dict_code_user)
        return np.array(y_pred)
