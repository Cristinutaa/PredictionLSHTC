from os import path
import time
import h2o
import numpy as np
import pandas as pd
from h2o.estimators import H2ORandomForestEstimator as RandomForest
from h2o import H2OFrame
from copy import deepcopy
from sklearn.metrics import accuracy_score
from util import get_config, timestamp, ensure_dir, setup_logger, force_locale_utf8, micro_recall, macro_recall, micro_precision, macro_precision, micro_f1, macro_f1
from classifiers import EncodedClassifier, MetaBinaryTreeClassifier, get_h2o_column_types, fit_h2o
import matplotlib.pyplot as plt

ENCODING_TYPES = ["random", "huffman", "balanced-tree", "meta-binary-tree-encoding"]
METRICS = {"accuracy": accuracy_score, "micro-recall": micro_recall, "macro-recall": macro_recall,
           "micro-precision": micro_precision, "macro-precision": macro_precision, "micro-f1": micro_f1,
           "macro-f1": macro_f1}


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
            classifier = EncodedClassifier(basic_classifier, encoding_type=classification)
        elif classification == "meta_binary_tree_classifier":
            classifier = MetaBinaryTreeClassifier(basic_classifier)
        elif classification == "standard-classifier":
            classifier = basic_classifier
        else:
            raise Exception("The Classification Method is not a valid one")
        start_time = time.time()
        if isinstance(classifier, h2o.estimators.H2OEstimator):
            classifier = fit_h2o(x_train, y_train, classifier)
        else:
            classifier.fit(x_train, y_train)
        train_time = time.time() - start_time
        if isinstance(classifier, h2o.estimators.H2OEstimator):
            column_types = get_h2o_column_types(x_test.columns)
            x_test = H2OFrame(x_test, column_types=column_types)
            prediction = classifier.predict(x_test)
            y_pred = np.concatenate(prediction['predict'].as_data_frame().values)
        else:
            y_pred = classifier.predict(x_test)
        prediction_time = time.time() - train_time - start_time
        # Calculate metrics
        for metric, f in METRICS.items():
            metrics_dict[metric][classification] = f(y_test, y_pred)
        training_time[classification] = train_time
        test_time[classification] = prediction_time

    return metrics_dict, training_time, test_time


def user_identification():
    cfg = get_config('h2o_different_nb_cls.cfg')
    h2o_cfg = get_config('h2o.cfg')

    h2o.init(nthreads=h2o_cfg.getint('h2o', 'nthreads'), max_mem_size=h2o_cfg.get('h2o', 'max_mem'))
    h2o.no_progress()
    h2o_seed = h2o_cfg.getint('h2o', 'seed')

    logger.info('intrusion_detection_synthetic')
    folder = cfg.get('data', 'path')
    if cfg.has_option('data', 'output_path'):
        output_folder = cfg.get('data', 'output_path')
        out_folder = path.join(output_folder, "H2O_" + timestamp())
    else:
        out_folder = path.join(folder, "H2O_" + timestamp())
    ensure_dir(out_folder)

    with open(path.join(out_folder, 'config.cfg'), 'w') as f:
        cfg.write(f)
    np.random.seed(cfg.getint('misc', 'random_seed'))

    # ignored columns, by name or prefix
    if cfg.has_option('data', 'ignored_columns'):
        ignored_columns = cfg.get('data', 'ignored_columns').split(',')
    else:
        ignored_columns = []
    if cfg.has_option('data', 'ignore_columns_starting_with'):
        ignore_columns_starting_with = cfg.get('data', 'ignore_columns_starting_with').split(',')
    else:
        ignore_columns_starting_with = []
    ignored_columns_reason = dict()

    n_folds = cfg.getint('data', 'n_folds')
    min_users = cfg.getint('data', 'min_users')
    max_users = cfg.getint('data', 'max_users')
    step = cfg.getint('data', 'step')
    train_frame = cfg.get('data', 'train')
    test_frame = cfg.get('data', 'test')

    logger.info('Out folder: %s' % out_folder)
    cluster_dir = path.join(out_folder, 'clusters')
    ensure_dir(cluster_dir)

    # print and check features on train set
    train_df = pd.read_csv(path.join(folder, train_frame))
    test_df = pd.read_csv(path.join(folder, test_frame))
    important_features = ['pc', 'http_count', 'session_length', 'session_end_hour', 'http_avg_duration',
                          'email_count', 'session_start_minute', 'session_start_hour', 'user']
    if set(important_features).issubset(set(train_df.columns)):
        print('OK')
        train_df = train_df[important_features]
        test_df = test_df[important_features]
    column_types = get_h2o_column_types(list(train_df))
    unique_users = np.unique(train_df['user'].unique())
    logger.info('total unique users: %d' % unique_users.shape[0])

    for c in column_types:
        if ignore_columns_starting_with and c.startswith(tuple(ignore_columns_starting_with)):
            ignored_columns.append(c)
            ignored_columns_reason[c] = 'ignored by prefix'

    ignored_columns.append('user')
    ignored_columns.append('is_anomaly')
    columns_to_keep = [i for i in list(train_df.columns) if i not in ignored_columns]

    # End of configuration
    # Preparation of the variables
    classification_types = ["meta_binary_tree_classifier", "random", "huffman", "balanced-tree",
                            "meta-binary-tree-encoding", "standard-classifier"]
    standard_dictionary = {}
    for i in classification_types:
        standard_dictionary[i] = []
    metrics_dict = {}
    for metric in METRICS:
        metrics_dict[metric] = deepcopy(standard_dictionary)
    training_time_dict = deepcopy(standard_dictionary)
    prediction_time_dict = deepcopy(standard_dictionary)
    number_of_users = [i for i in range(min_users, max_users + 1, step)]
    logger.info("number_of_users = " + str(number_of_users))
    for n_users in number_of_users:
        logger.info("____________________________________")
        logger.info("DATA FOR %d CLASSES" % n_users)
        n_metrics_dict = {}
        for metric in METRICS:
            n_metrics_dict[metric] = deepcopy(standard_dictionary)
        n_train_time = deepcopy(standard_dictionary)
        n_predict_time = deepcopy(standard_dictionary)
        rf = RandomForest(seed=h2o_seed, ntrees=cfg.getint('random_forest', 'ntrees'),
                          max_depth=cfg.getint('random_forest', 'max_depth'),
                          categorical_encoding=cfg.get('random_forest', 'categorical_encoding'),
                          nbins_cats=cfg.getint('random_forest', 'nbins_cats'),
                          histogram_type=cfg.get('random_forest', 'histogram_type'))

        for i in range(n_folds):
            logger.info("++++++++++++++++++++++++++++++++++")
            logger.info("Fold %d" % (i + 1))
            fold_users = np.random.choice(unique_users, n_users, replace=False)
            x_train_fold = train_df.loc[train_df['user'].isin(fold_users)].reset_index(drop=True)
            x_test_fold = test_df.loc[test_df['user'].isin(fold_users)].reset_index(drop=True)
            y_train = x_train_fold['user']
            y_test = x_test_fold['user']
            x_train_fold = x_train_fold[columns_to_keep]
            x_test_fold = x_test_fold[columns_to_keep]
            while len(y_test) == 0:
                fold_users = np.random.choice(unique_users, n_users, replace=False)
                x_train_fold = train_df.loc[train_df['user'].isin(fold_users)].reset_index(drop=True)
                x_test_fold = test_df.loc[test_df['user'].isin(fold_users)].reset_index(drop=True)
                y_train = x_train_fold['user']
                y_test = x_test_fold['user']
                x_train_fold = x_train_fold[columns_to_keep]
                x_test_fold = x_test_fold[columns_to_keep]

            temp_metrics_dict, train, test = fold_prediction_result(x_train_fold, y_train, x_test_fold,
                                                                    y_test, classification_types, rf)
            for classification in classification_types:
                for metric in METRICS:
                    n_metrics_dict[metric][classification].append(temp_metrics_dict[metric][classification])
                n_train_time[classification].append(train[classification])
                n_predict_time[classification].append(test[classification])
            h2o.remove_all()
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
            ax.plot(number_of_users, metrics_dict[score][classification])
            plt.title("Classification %s for %s" % (score, classification))
            plt.xlabel("Number of classes")
            plt.ylabel(score)
        plt.savefig(path.join(out_folder, classification))
        plt.close(fig)

    fig = plt.figure(figsize=(15, 30))
    for i, score in enumerate(metrics_dict.keys()):
        ax = fig.add_subplot(len(set(metrics_dict))//2+1, 2, i + 1)
        for classification in classification_types:
            ax.plot(number_of_users, metrics_dict[score][classification])
        plt.title("Classification %s" % score)
        plt.legend(classification_types, loc="upper right")
        plt.xlabel("Number of classes")
        plt.ylabel(score)
    plt.savefig(path.join(out_folder, "all"))
    plt.close(fig)


def main():
    force_locale_utf8()
    user_identification()
    return 0


if __name__ == '__main__':
    logger = setup_logger('user_identification')
    main()
