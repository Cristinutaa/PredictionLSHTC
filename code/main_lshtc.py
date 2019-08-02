import time
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from scipy import sparse
from numpy import shape
from copy import deepcopy
from code.classifiers import EncodedClassifier, MagicTreeClassifier
from code.util import setup_logger
import matplotlib.pyplot as plt


def get_dict(x, y):
    row_dict = {}
    for i, c in enumerate(y):
        if c in row_dict:
            row_dict[c].append(x[i])
        else:
            row_dict[c] = [x[i]]
    return row_dict


def get_data_for_classes(row_dict, y, classes):
    row_dict = deepcopy(row_dict)
    y_filtered = [i for i in y if i in classes]
    x = sparse.vstack([row_dict[i].pop(0) for i in y_filtered])
    return x, y_filtered


def prediction_results(y_train, y_test, classifier, n_users, unique_users, number_iterations, row_dict_train,
                       row_dict_test):
    micro_acc = 0
    macro_acc = 0
    training_time = 0
    test_time = 0
    for i in range(number_iterations):
        # Prepare the data
        fold_users = np.random.choice(unique_users, n_users, replace=False)
        x_train_fold, y_train_fold = get_data_for_classes(row_dict_train, y_train, fold_users)
        x_test_fold, y_test_fold = get_data_for_classes(row_dict_test, y_test, fold_users)
        # Prediction
        start_time = time.time()
        classifier.fit(x_train_fold, y_train_fold)
        train_time = time.time() - start_time
        y_pred = classifier.predict(x_test_fold)
        prediction_time = time.time() - start_time - train_time
        # Calculate metrics
        micro_acc += accuracy_score(y_test_fold, y_pred)
        macro_acc += balanced_accuracy_score(y_test_fold, y_pred)
        training_time += train_time
        test_time += prediction_time
    # Average metrics
    micro_acc = micro_acc / number_iterations
    macro_acc = macro_acc / number_iterations
    training_time = training_time / number_iterations
    test_time = test_time / number_iterations
    # Print metrics
    logger.info("Average Micro-Accuracy %2.4f" % micro_acc)
    logger.info("Average Macro-Accuracy %2.4f" % macro_acc)
    logger.info("Average train time: %d min %d s" % (training_time // 60, training_time % 60))
    logger.info("Average prediction time: %d min %d s" % (test_time // 60, test_time % 60))
    return micro_acc, macro_acc, training_time, test_time


def main():
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
    min_users = 500
    max_users = 1000
    step = 100

    micro_accuracies = {'meta_binary_tree_classifier': [], "random": [], "huffman": [], "balanced-tree": [],
                        "meta-binary-tree-encoding": [], "one-vs-all": []}
    macro_accuracies = {'meta_binary_tree_classifier': [], "random": [], "huffman": [], "balanced-tree": [],
                        "meta-binary-tree-encoding": [], "one-vs-all": []}
    training_time_dict = {'meta_binary_tree_classifier': [], "random": [], "huffman": [], "balanced-tree": [],
                          "meta-binary-tree-encoding": [], "one-vs-all": []}
    prediction_time_dict = {'meta_binary_tree_classifier': [], "random": [], "huffman": [], "balanced-tree": [],
                            "meta-binary-tree-encoding": [], "one-vs-all": []}
    number_of_users = [i for i in range(min_users, max_users, step)]
    logger.info("number_of_users = " + str(number_of_users))

    for n_users in number_of_users:
        logger.info("DATA FOR %d CLASSES" % n_users)

        logger.info("Meta-Binary Tree Classifier")
        cls = MagicTreeClassifier(estimator=SVC(class_weight="balanced", random_state=1, gamma="auto"))
        micro_acc, macro_acc, train_time, prediction_time = prediction_results(y_train, y_test, cls, n_users,
                                                                               unique_users, number_iterations,
                                                                               row_dict_train, row_dict_test)
        micro_accuracies['meta_binary_tree_classifier'].append(micro_acc)
        macro_accuracies['meta_binary_tree_classifier'].append(macro_acc)
        prediction_time_dict['meta_binary_tree_classifier'].append(prediction_time)
        training_time_dict['meta_binary_tree_classifier'].append(train_time)
        logger.info("*************")

        for encoding in ["random", "huffman", "balanced-tree", "meta-binary-tree-encoding"]:
            logger.info("Classification using encoding %s" % encoding)
            cls = EncodedClassifier(SVC(class_weight="balanced", random_state=1, gamma="auto"), encoding_type=encoding)
            micro_acc, macro_acc, train_time, prediction_time = prediction_results(y_train, y_test, cls, n_users,
                                                                                   unique_users, number_iterations,
                                                                                   row_dict_train, row_dict_test)
            micro_accuracies[encoding].append(micro_acc)
            macro_accuracies[encoding].append(macro_acc)
            prediction_time_dict[encoding].append(prediction_time)
            training_time_dict[encoding].append(train_time)
            logger.info("*************")

        logger.info("Standard Classifier")
        cls = SVC(class_weight="balanced", random_state=1, gamma="auto")
        micro_acc, macro_acc, train_time, prediction_time = prediction_results(y_train, y_test, cls, n_users,
                                                                               unique_users, number_iterations,
                                                                               row_dict_train, row_dict_test)
        micro_accuracies['one-vs-all'].append(micro_acc)
        macro_accuracies['one-vs-all'].append(macro_acc)
        prediction_time_dict['one-vs-all'].append(prediction_time)
        training_time_dict['one-vs-all'].append(train_time)
        logger.info("*************")
    logger.info("micro_accuracies = " + str(micro_accuracies))
    logger.info("macro_accuracies = " + str(macro_accuracies))
    logger.info("training_time = " + str(training_time_dict))
    logger.info("prediction_time = " + str(prediction_time_dict))

    # Plot generation
    scores_dict = {"micro accuracy": micro_accuracies, "macro accuracy": macro_accuracies,
                   "training time (seconds)": training_time_dict, "prediction time (seconds)": prediction_time_dict}
    classification_types = ["meta_binary_tree_classifier", "random", "huffman", "balanced-tree",
                            "meta-binary-tree-encoding", "one-vs-all"]
    for classification in classification_types:
        fig = plt.figure(figsize=(15, 15))
        for i, score in enumerate(scores_dict.keys()):
            ax = fig.add_subplot(2, 2, i + 1)
            ax.plot(number_of_users, scores_dict[score][classification])
            plt.title("Classification %s for %s" % (score, classification))
            plt.xlabel("Number of classes")
            plt.ylabel(score)
        plt.savefig("../images/LSHTC_" + classification)
        plt.close(fig)

    fig = plt.figure(figsize=(15, 15))
    for i, score in enumerate(scores_dict.keys()):
        ax = fig.add_subplot(2, 2, i + 1)
        for classification in classification_types:
            ax.plot(number_of_users, scores_dict[score][classification])
        plt.title("Classification %s" % score)
        plt.legend(classification_types, loc="upper right")
        plt.xlabel("Number of classes")
        plt.ylabel(score)
    plt.savefig("../images/LSHTC_all")
    plt.close(fig)


if __name__ == '__main__':
    logger = setup_logger('lshtc')
    main()
