import time
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy import sparse
from numpy import shape
from copy import deepcopy
from classifiers import EncodedClassifier, MetaBinaryTreeClassifier
from util import setup_logger, force_locale_utf8, micro_precision, micro_recall, macro_precision, macro_recall, micro_f1, macro_f1
import matplotlib.pyplot as plt

ENCODING_TYPES = ["random", "huffman", "balanced-tree", "meta-binary-tree-encoding"]
METRICS = {"accuracy": accuracy_score, "micro-recall": micro_recall, "macro-recall": macro_recall,
           "micro-precision": micro_precision, "macro-precision": macro_precision, "micro-f1": micro_f1,
           "macro-f1": macro_f1}


def get_dict(x, y):
    """
    Generates dictionaries with data (rows) from x for every unique class from y
    :param x: the data
    :param y: the list of classes for data in y
    :return: row_dict - dictionary with the class as key and the list of rows corresponding to this class as value
    """
    row_dict = {}
    for i, c in enumerate(y):
        if c in row_dict:
            row_dict[c].append(x[i])
        else:
            row_dict[c] = [x[i]]
    return row_dict


def get_data_for_classes(row_dict, y, classes):
    """
    Recompose X from row_dict and filter y only for the data corresponding for classes.
    Could be used as data for training a classifier
    :param row_dict: the dictionary with the list of rows corresponding for each class
    :param y: the list of classes from which only the one from the variable classes will be selected
    :param classes: the classes to be kept
    :return: x - the data for the classes from y_filtered
             y_filtered - y with only the classes from classes
    """
    row_dict = deepcopy(row_dict)
    y_filtered = [i for i in y if i in classes]
    x = sparse.vstack([row_dict[i].pop(0) for i in y_filtered])
    return x, y_filtered


def fold_prediction_result(x_train, y_train, x_test, y_test, classification_types, basic_classifier):
    """
    The training and prediction for one fold for all the types of classifiers indicated in classification_types
    :param x_train: the training data
    :param y_train: the training classes
    :param x_test: the testing data
    :param y_test: the testing classes
    :param classification_types: the classification types to be considered
    :param basic_classifier: the basic classifier to be used either independently or for the meta classifiers
    :return: metrics_dict - dictionary containing a dictionary for every metric with data for every classifier
             training_time - dictionary with training time in seconds for every classification type
             test_time - dictionary with testing time in seconds for every classification type
    """
    metrics_dict = {}
    for metric in METRICS:
        metrics_dict[metric] = {}
    training_time = {}
    test_time = {}
    for classification in classification_types:
        # logger.info("*****************************")
        logger.info(classification)
        if classification in ENCODING_TYPES:
            classifier = EncodedClassifier(basic_classifier, encoding_type=classification, n_jobs=8)
        elif classification == "meta_binary_tree_classifier":
            classifier = MetaBinaryTreeClassifier(basic_classifier)
        elif classification == "one-vs-all":
            classifier = basic_classifier
        else:
            raise Exception("The Classification Method is not a valid one")
        start_time = time.time()
        classifier.fit(x_train, y_train)
        train_time = time.time() - start_time
        y_pred = classifier.predict(x_test)
        prediction_time = time.time() - start_time - train_time
        # Calculate metrics
        for metric, f in METRICS.items():
            metrics_dict[metric][classification] = f(y_test, y_pred)
        training_time[classification] = train_time
        test_time[classification] = prediction_time

    return metrics_dict, training_time, test_time


def main():
    # Configuration
    train_file = "C:/Users/Cristina/PhD Track/Data/LSHTC/LSHTC1/train_corrected.txt"
    test_file = "C:/Users/Cristina/PhD Track/Data/LSHTC/LSHTC1/validation_corrected.txt"

    x_train, y_train = load_svmlight_file(train_file)
    x_test, y_test = load_svmlight_file(test_file)

    if shape(x_train)[1] > shape(x_test)[1]:
        x_test, y_test = load_svmlight_file(test_file, n_features=shape(x_train)[1])
    else:
        x_train, y_train = load_svmlight_file(train_file, n_features=shape(x_test)[1])

    y_train = [str(int(i)) for i in y_train]
    y_test = [str(int(i)) for i in y_test]

    row_dict_train = get_dict(x_train, y_train)
    row_dict_test = get_dict(x_test, y_test)
    unique_users = np.unique(np.array(y_train))
    number_iterations = 5
    min_classes = 50
    max_classes = 3050
    step = 100
    classification_types = ["meta_binary_tree_classifier", "random", "huffman", "balanced-tree",
                            "meta-binary-tree-encoding", "one-vs-all"]
    # End of configuration
    # Preparation of the variables
    standard_dictionary = {}
    for i in classification_types:
        standard_dictionary[i] = []
    metrics_dict = {}
    for metric in METRICS:
        metrics_dict[metric] = deepcopy(standard_dictionary)
    training_time_dict = deepcopy(standard_dictionary)
    prediction_time_dict = deepcopy(standard_dictionary)
    number_of_classes = [i for i in range(min_classes, max_classes + 1, step)]
    logger.info("number_of_classes = " + str(number_of_classes))
    estimator = SVC(class_weight="balanced", random_state=1, gamma="auto")

    # Classification and metrics calculation
    for n_users in number_of_classes:
        logger.info("____________________________________")
        logger.info("DATA FOR %d CLASSES" % n_users)
        # Dictionaries for data for this number of users
        n_metrics_dict = {}
        for metric in METRICS:
            n_metrics_dict[metric] = deepcopy(standard_dictionary)
        n_train_time = deepcopy(standard_dictionary)
        n_predict_time = deepcopy(standard_dictionary)
        # Folds
        for i in range(number_iterations):
            logger.info("++++++++++++++++++++++++++++++++++")
            logger.info("Fold %d" % (i + 1))
            fold_users = np.random.choice(unique_users, n_users, replace=False)
            x_train_fold, y_train_fold = get_data_for_classes(row_dict_train, y_train, fold_users)
            x_test_fold, y_test_fold = get_data_for_classes(row_dict_test, y_test, fold_users)
            while len(y_test) == 0:
                fold_users = np.random.choice(unique_users, n_users, replace=False)
                x_train_fold, y_train_fold = get_data_for_classes(row_dict_train, y_train, fold_users)
                x_test_fold, y_test_fold = get_data_for_classes(row_dict_test, y_test, fold_users)
            temp_metrics_dict, train, test = fold_prediction_result(x_train_fold, y_train_fold, x_test_fold, y_test_fold,
                                                               classification_types, estimator)
            for classification in classification_types:
                for metric in METRICS:
                    n_metrics_dict[metric][classification].append(temp_metrics_dict[metric][classification])
                n_train_time[classification].append(train[classification])
                n_predict_time[classification].append(test[classification])

        # Aggregation of the fold results and append to the global dictionaries
        for classification in classification_types:
            logger.info("***___***___***___***___***___")
            logger.info("Average data for %s" % classification)
            for metric in METRICS:
                avg = np.average(n_metrics_dict[metric][classification])
                metrics_dict[metric][classification].append(avg)
                logger.info("Average %s for %s for %d users: %2.2f" % (metric, classification, n_users, avg))
            avg = np.average(n_train_time[classification])
            training_time_dict[classification].append(avg)
            logger.info("Average training time for %s for %d users : %d min %d s " % (
                classification, n_users, avg // 60, avg % 60))
            avg = np.average(n_predict_time[classification])
            prediction_time_dict[classification].append(avg)
            logger.info("Average prediction time for %s for %d users : %d min %d s " % (
                classification, n_users, avg // 60, avg % 60))

    for metric in METRICS:
        logger.info("%s = %s" % (metric, metrics_dict[metric]))
    logger.info("training_time = " + str(training_time_dict))
    logger.info("prediction_time = " + str(prediction_time_dict))
    metrics_dict["training time (seconds)"] = training_time_dict
    metrics_dict["prediction time (seconds)"] = prediction_time_dict

    # Plot generation
    for classification in classification_types:
        fig = plt.figure(figsize=(15, 30))
        for i, score in enumerate(metrics_dict.keys()):
            ax = fig.add_subplot(len(set(metrics_dict))//2+1, 2, i + 1)
            ax.plot(number_of_classes, metrics_dict[score][classification])
            plt.title("Classification %s for %s" % (score, classification))
            plt.xlabel("Number of classes")
            plt.ylabel(score)
        plt.savefig("../images/LSHTC_" + classification)
        plt.close(fig)

    fig = plt.figure(figsize=(15, 30))
    for i, score in enumerate(metrics_dict.keys()):
        ax = fig.add_subplot(len(set(metrics_dict))//2+1, 2, i + 1)
        for classification in classification_types:
            ax.plot(number_of_classes, metrics_dict[score][classification])
        plt.title("Classification %s" % score)
        plt.legend(classification_types, loc="upper right")
        plt.xlabel("Number of classes")
        plt.ylabel(score)
    plt.savefig("../images/LSHTC_all")
    plt.close(fig)


if __name__ == '__main__':
    force_locale_utf8()
    logger = setup_logger('lshtc')
    main()
