import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

VALID_STATS = {"micro_acc", "macro_acc", "acc_0_count", "acc_sup_micro", "acc_inf_micro", "acc_distribution"}


def histogram_data(y, file_name, suffix):
    """
    Builds the histograms with the frequency of classes in the data y
    :param y: the data with classes
    :param file_name: the name of the file where the images should be saved.
           The file is saved in the image directory of the same folder
    :param suffix: The suffix to be put in the titles of histograms "Class frequency" + suffix
    """
    _, freq = np.unique(y, return_counts=True)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, )
    counts, values, patches = ax.hist(freq, bins=np.arange(freq.min(), freq.max() + 1))
    plt.title("Class frequency " + suffix)
    plt.xlabel("The frequency of classes")
    plt.ylabel("Number of classes")
    plt.savefig("images/" + file_name)
    plt.show()
    plt.close(fig)

    n = 20
    freq = list(freq)
    while sum(counts[len(counts) - n:]) < 10:
        values_del = values[len(counts) - n:]
        freq = list(filter(lambda a: a not in values_del, freq))
        n += 20
    freq = np.array(freq)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, )
    ax.hist(freq, bins=np.arange(freq.min(), freq.max() + 1))
    plt.title("Class frequency " + suffix + " zoomed")
    plt.xlabel("The frequency of classes")
    plt.ylabel("Number of classes")
    plt.savefig("images/" + file_name + "_zoomed")
    plt.show()
    plt.close(fig)


def get_class_accuracy(y_pred, y_test):
    """
    Calculates a list with the accuracy of the prediction of each class
    :param y_pred: the values that were predicted
    :param y_test: the correct values
    :return: a list with the accuracy of the prediction of each class
    """
    cm = confusion_matrix(y_test, y_pred, labels=list(set(y_test)))
    total = [[1] if x == [0] else x for x in cm.sum(axis=1)[:, np.newaxis]]
    cm = cm.astype('float') / total
    accuracies = np.array(cm.diagonal())
    return accuracies


def statistics(y_train=[], y_pred=[], y_test=[], show_hist=True, file_name="", stats=VALID_STATS):
    """
    Calculates the statistics for the data
    :param y_train: the class list of the training set (used for data statistics)
    :param y_pred: the list of predicted classes (used for statistics on prediction)
    :param y_test: the correct class list of the test set (used for data statistics and statistics on prediction)
    :param show_hist: boolean value that indicated if data histograms should be created
    :param file_name: the name of the file where the histograms should be created
    :param stats: which statistics to be printed
    """
    if not stats.issubset(VALID_STATS):
        raise ValueError("The stats set contains unknown statistics")
    print("Statistics on data")
    print("-----------")
    if show_hist:
        print("show_hist: True")
        y = np.append(y_train, y_test)
        histogram_data(y, file_name + "train_test", "train + test")
        # histogram_data(np.array(y_train), file_name + "train", "train")
        # histogram_data(np.array(y_test), file_name + "test", "test")

    micro_accuracy = accuracy_score(y_test, y_pred) * 100
    accuracies = get_class_accuracy(y_pred, y_test)

    if "micro_acc" in stats:
        print("Accuracy: %2.2f %%" % micro_accuracy)

    if "macro_acc" in stats:
        print("Average accuracy per class %2.2f %%" % (accuracies.mean() * 100))

    # print("Standard deviation of accuracy per class %2.2f %%" % (np.std(accuracies) * 100))

    if "acc_inf_micro" in stats:
        print("Number of classes with accuracy < to %2.2f %% - %d" % (
            micro_accuracy, (100 * accuracies < micro_accuracy).sum()))

    if "acc_0_count" in stats:
        print("Number of classes with accuracy 0 - %d" % (100 * accuracies == 0).sum())

    if "acc_sup_micro" in stats:
        print("Number of classes with accuracy > to %2.2f %% - %d" % (
            micro_accuracy, (100 * accuracies > micro_accuracy).sum()))
    if "acc_distribution":
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, )
        ax.hist(100 * accuracies, bins=100)
        plt.title("Accuracies distribution")
        plt.xlabel("Accuracy value")
        plt.ylabel("Number of classes")
        plt.show()
