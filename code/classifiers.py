import random
import pandas as pd
import numpy as np
import h2o
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from scipy import sparse
from h2o import H2OFrame
from h2o.estimators import H2ORandomForestEstimator as RandomForest

from encoding_decoding import create_balanced_tree_encoding, create_encoding_huffman, \
    create_encoding_random, decode_users, encode_users, encode_classes, BinaryNode


def get_h2o_column_types(column_names):
    types = dict()
    for c in column_names:
        if c in {'session_begin', 'session_start', 'session_end'}:
            types[c] = 'time'
        elif c in {'n_events', 'ip_long', 'ip_lat', 'logons'}:
            types[c] = 'numeric'
        elif c in {'country_match', 'user_country_match', 'id', 'ip', 'ip_city', 'ip_region', 'ip_country', 'user',
                   'user_country', 'modified_user', 'anomaly_scenario', 'pc'}:
            types[c] = 'enum'
        elif c.startswith(('app_count_', 'auth_', 'failed_auth_', 'bigram_', 'timedelta_',
                           'transition_delta_', 'since_', 'session_', 'hour_events_', 'app_duration_',
                           'device_', 'email_', 'file_', 'http_')):
            types[c] = 'numeric'
        elif c.startswith('is_anomaly'):
            types[c] = 'enum'
        else:
            raise ValueError('Unknown feature:', c)

    return types


def fit_h2o(x_train, y_train, estimator):
    parameters = estimator._parms
    estimator_type = estimator.__class__
    current_estimator = estimator_type()
    current_estimator._parms = parameters
    column_types_x = get_h2o_column_types(x_train.columns)
    x_train = H2OFrame(x_train, column_types=column_types_x)
    y_train = H2OFrame(list(y_train), column_types=['enum'])
    training_frame = x_train.cbind(y_train) if y_train is not None else x_train
    x_train = x_train.names
    y_train = y_train.names[0]
    current_estimator.train(x_train, y_train, training_frame)
    return current_estimator


def classify(x_train, y_train, estimator, x_test):
    """
    Make the classification and provide the result for the given estimator.
    For the H2O library, the transformation in H2oFrame is integrated
    :param x_train: the dataset for training
    :param y_train: the classes for training
    :param estimator: the estimator to be considered
    :param x_test: the dataset for testing
    :return: - the prediction for the x_test
             - the trained estimator
    """
    if isinstance(estimator, h2o.estimators.H2OEstimator):
        current_estimator = fit_h2o(x_train, y_train, estimator)
        column_types_x = get_h2o_column_types(x_test.columns)
        x_test = H2OFrame(x_test, column_types=column_types_x)
        prediction = current_estimator.predict(x_test)
        return np.concatenate(prediction['predict'].as_data_frame().values), current_estimator
    else:
        current_estimator = clone(estimator)
        current_estimator.fit(x_train, y_train)
        return current_estimator.predict(x_test), current_estimator


# BinaryNode with classifier and classes per BinaryNode
class ClassificationBinaryNode(BinaryNode):
    def __init__(self, classes, classifier=None, left_node=None, right_node=None, level=0):
        BinaryNode.__init__(self, left_node, right_node, level)
        self.classes = classes
        self.classifier = classifier


def two_set_prediction(set1, x_train, y_train, x_test, y_test, estimator):
    """
    return the prediction result of attributing the classes to two bins 0 and 1
    :param set1: One of the bins of classes
    :param x_train: the train data
    :param y_train: the train classes
    :param x_test: the test data
    :param y_test: the test classes
    :param estimator: the estimator model to be considered
    :return: current estimator - the estimator used for the prediction
             predicted - the predicted classes (Iterable)
             accuracy - the accuracy  got for the prediction
    """
    y_train_1 = [0 if y in set1 else 1 for y in y_train]
    y_test_1 = [0 if y in set1 else 1 for y in y_test]
    predicted, current_estimator = classify(x_train, y_train_1, estimator, x_test)
    acc = accuracy_score(predicted, y_test_1)
    return current_estimator, predicted, acc


def redo_repartition(predicted, y_test):
    """
    Redo the repartition in bins considering the given prediction for every class
    :param predicted: the list of predicted classes (expected two classes: 0 and 1)
    :param y_test: the list of the actual classes
    :return: y_1 - the first bin/set of classes
             y_2 - the second bin/set of classes
    """
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
    return y_1, y_2


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
    current_estimator, predicted, acc = two_set_prediction(y_1, x_train, y_train, x_test, y_test, estimator)
    max_acc = acc
    best_sets = [y_1, y_2]
    best_est = current_estimator
    count = 1
    # Loop that tries to improve the repartition and classify again in order to improve the results
    while acc < target_acc and count < max_iterations:
        count += 1
        y_1, y_2 = redo_repartition(predicted, y_test)
        current_estimator, predicted, acc = two_set_prediction(y_1, x_train, y_train, x_test, y_test, estimator)
        if acc > max_acc:
            max_acc = acc
            best_sets = [y_1, y_2]
            best_est = current_estimator
    return best_est, best_sets[0], best_sets[1]


def split_x(x, y, y_1, y_2):
    """
    Split the x data in two structures: one containing the data for classes from y_1,
                                        another containing the data for classes from y_2
    :param x: the data to be separated
    :param y: the class list for the given data
    :param y_1: the first set of classes
    :param y_2: the second set of classes
    :return: x_1, x_2 - the two data partitions
    """
    if sparse.issparse(x):
        row_dict_train = {}
        for i, c in enumerate(y):
            if c in row_dict_train:
                row_dict_train[c].append(x[i])
            else:
                row_dict_train[c] = [x[i]]
        x_1 = sparse.vstack([row_dict_train[i].pop(0) for i in y_1])
        x_2 = sparse.vstack([row_dict_train[i].pop(0) for i in y_2])
        return x_1, x_2
    elif isinstance(x, pd.DataFrame):
        row_dict_train = {}
        for i, c in enumerate(y):
            if c in row_dict_train:
                row_dict_train[c].append(i)
            else:
                row_dict_train[c] = [i]
        x_1 = x.iloc[[row_dict_train[i].pop() for i in y_1]].reset_index(drop=True)
        x_2 = x.iloc[[row_dict_train[i].pop() for i in y_2]].reset_index(drop=True)
        return x_1, x_2


def node_train(node, x_train, y_train, estimator):
    """
    Train the tree with node as root, fixes the classes, the classifier and the level of the nodes
    :param node: the root node
    :param x_train: data for training
    :param y_train: classes for training
    :param estimator: the model of the estimator to be considered
    """
    if len(node.classes) > 1:
        classifier, y_1, y_2 = train_classifier(x_train, y_train, x_train, y_train, estimator)
        node.classifier = classifier
        y_1_train = [y for y in y_train if y in y_1]
        y_2_train = [y for y in y_train if y in y_2]
        node.left_node = ClassificationBinaryNode(y_1)
        x_1_train, x_2_train = split_x(x_train, y_train, y_1_train, y_2_train)
        node_train(node.left_node, x_1_train, y_1_train, estimator)
        node.right_node = ClassificationBinaryNode(y_2)
        node_train(node.right_node, x_2_train, y_2_train, estimator)
        node.change_level(max(node.right_node.level, node.left_node.level) + 1)
    else:
        node.change_level(0)


def create_meta_binary_tree_encoding(x_train, y_train, estimator):
    """
    Create the encoding for the classes for y train using the Meta Binary Tree method
    :param x_train: the data for training
    :param y_train: the classes for training
    :param estimator: the model of the estimator to be considered
    :return: class_to_codes - dictionary with classes as keys and the corresponding code as value
             codes_to_classes - dictionary with codes as keys and the corresponding class as value
    """
    root = ClassificationBinaryNode(set(y_train))
    node_train(root, x_train, y_train, estimator)
    class_to_codes, codes_to_class = encode_classes(root, "", root.level, "classes", True)
    return class_to_codes, codes_to_class, root.level


def predict_row(row, node):
    if node.classifier:
        if isinstance(node.classifier, h2o.estimators.H2OEstimator):
            if not isinstance(row, H2OFrame):
                column_types_row = get_h2o_column_types(row.columns)
                row = H2OFrame(row, column_types=column_types_row)
            prediction = node.classifier.predict(row)
            prediction = np.concatenate(prediction['predict'].as_data_frame().values)
        else:
            prediction = node.classifier.predict(row)
        if prediction[0] == 0:
            prediction = predict_row(row, node.left_node)
        else:
            prediction = predict_row(row, node.right_node)
    else:
        return list(node.classes)[0]

    return prediction


def predict_dataframe(data, node):
    if node.classifier:
        if isinstance(node.classifier, h2o.estimators.H2OEstimator):
            if not isinstance(data, H2OFrame):
                column_types = get_h2o_column_types(data.columns)
                data_h2o = H2OFrame(data, column_types=column_types)
            prediction = node.classifier.predict(data_h2o)
            if len(prediction['predict'].as_data_frame().values) == 0:
                prediction = np.array([])
            else:
                prediction = np.concatenate(prediction['predict'].as_data_frame().values)
        else:
            prediction = node.classifier.predict(data)
        data_right = data.iloc[[i for i in range(len(prediction)) if prediction[i] == 1]]
        data_left = data.iloc[[i for i in range(len(prediction)) if prediction[i] == 0]]
        prediction_left = predict_dataframe(data_left, node.left_node)
        prediction_right = predict_dataframe(data_right, node.right_node)
        return sorted(prediction_left + prediction_right)
    else:
        return [(i, list(node.classes)[0]) for i in list(data.index)]


def predict_meta_binary_tree(x_test, root):
    number_rows = x_test.shape[0]
    if sparse.issparse(x_test):
        prediction = [predict_row(x_test.getrow(i), root) for i in range(number_rows)]
    elif isinstance(x_test, pd.DataFrame):
        prediction = predict_dataframe(x_test, root)
        prediction = [el[1] for el in prediction]
    return prediction


def fit_single(estimator, x, y):
    """
    Train the estimator with X,y data
    :param estimator: the estimator model to be considered
    :param x: the data for training
    :param y: the class list for training
    :return: the trained clone of the classifier
    """
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
    def __encode(self, x, y):
        if self.encoding_type == "huffman":
            y = pd.Series(y)
            self.dict_user_code, self.dict_code_user, self.encoding_length = create_encoding_huffman(y)
        elif self.encoding_type == "balanced-tree":
            y = pd.Series(y)
            self.dict_user_code, self.dict_code_user, self.encoding_length = create_balanced_tree_encoding(y)
        elif self.encoding_type == "random":
            self.dict_user_code, self.dict_code_user, self.encoding_length = create_encoding_random(set(y))
        elif self.encoding_type == "meta-binary-tree-encoding":
            self.dict_user_code, self.dict_code_user, self.encoding_length = create_meta_binary_tree_encoding(x, y,
                                                                                                              self.estimator)
        encoded_train = encode_users(y, self.dict_user_code, self.encoding_length)
        return encoded_train

    def fit(self, x, y=None):
        y = self.__encode(x, y)
        if isinstance(self.estimator, h2o.estimators.H2OEstimator):
            self.estimators_ = [fit_h2o(x, y[column], self.estimator) for column in y]
            return self
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(fit_single)(self.estimator, x, y[column]) for column in y)
        return self

    def predict(self, x):
        if isinstance(self.estimator, h2o.estimators.H2OEstimator):
            column_types_x = get_h2o_column_types(x.columns)
            x = H2OFrame(x, column_types=column_types_x)
            results = pd.DataFrame(index=range(len(x)))
            i = 0
            for estimator in self.estimators_:
                predictions = estimator.predict(x)
                results[i] = predictions['predict'].as_data_frame().values
                i += 1
        else:
            results = np.array([estimator.predict(x) for estimator in self.estimators_]).T
            results = pd.DataFrame(results)
        y_pred = decode_users(results, self.dict_code_user)
        return np.array(y_pred)


class MetaBinaryTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, n_jobs=1):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.estimators_ = None
        self.root = None

    def fit(self, x, y=None):
        self.root = ClassificationBinaryNode(set(y))
        node_train(self.root, x, y, self.estimator)
        return self

    def predict(self, x):
        return np.array(predict_meta_binary_tree(x, self.root))
