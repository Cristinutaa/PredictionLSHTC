from os import path
import time
import h2o
import numpy as np
import pandas as pd
from h2o.estimators import H2ORandomForestEstimator as RandomForest
from h2o import H2OFrame
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from code.util import get_config, timestamp, ensure_dir, setup_logger
from code.classifiers import EncodedClassifier, MagicTreeClassifier, get_h2o_column_types, fit_h2o
import matplotlib.pyplot as plt


def prediction_results(x_train, x_test, classifier, n_users, unique_users, number_iterations, columns_to_keep):
    micro_acc = 0
    macro_acc = 0
    training_time = 0
    test_time = 0
    for i in range(number_iterations):
        # Prepare the data
        fold_users = np.random.choice(unique_users, n_users, replace=False)
        x_train_fold = x_train.loc[x_train['user'].isin(fold_users)].reset_index(drop=True)
        x_test_fold = x_test.loc[x_test['user'].isin(fold_users)].reset_index(drop=True)
        y_train = x_train_fold['user']
        y_test = x_test_fold['user']
        x_train_fold = x_train_fold[columns_to_keep]
        x_test_fold = x_test_fold[columns_to_keep]
        # Prediction
        start_time = time.time()
        if isinstance(classifier, h2o.estimators.H2OEstimator):
            classifier = fit_h2o(x_train_fold, y_train, classifier)
        else:
            classifier.fit(x_train_fold, y_train)
        train_time = time.time() - start_time
        if isinstance(classifier, h2o.estimators.H2OEstimator):
            column_types = get_h2o_column_types(x_test_fold.columns)
            x_test_fold = H2OFrame(x_test_fold, column_types=column_types)
            prediction = classifier.predict(x_test_fold)
            y_pred = np.concatenate(prediction['predict'].as_data_frame().values)
        else:
            y_pred = classifier.predict(x_test_fold)
        prediction_time = time.time() - start_time - train_time
        # Calculate metrics
        micro_acc += accuracy_score(y_test, y_pred)
        macro_acc += balanced_accuracy_score(y_test, y_pred)
        training_time += train_time
        test_time += prediction_time
        h2o.remove_all()
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


def user_identification():
    cfg = get_config('h2o_different_nb_cls.cfg')
    h2o_cfg = get_config('h2o.cfg')

    h2o.init(nthreads=h2o_cfg.getint('h2o', 'nthreads'), max_mem_size=h2o_cfg.get('h2o', 'max_mem'))
    h2o.no_progress()
    h2o_seed = h2o_cfg.getint('h2o', 'seed')

    logger.info('intrusion_detection_synthetic')
    folder = cfg.get('data', 'path')
    out_folder = path.join(folder, timestamp())
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

    # Models testing
    micro_accuracies = {'meta_binary_tree_classifier': [], "random": [], "huffman": [], "balanced-tree": [],
                        "meta-binary-tree-encoding": [], "standard-classifier": []}
    macro_accuracies = {'meta_binary_tree_classifier': [], "random": [], "huffman": [], "balanced-tree": [],
                        "meta-binary-tree-encoding": [], "standard-classifier": []}
    training_time_dict = {'meta_binary_tree_classifier': [], "random": [], "huffman": [], "balanced-tree": [],
                          "meta-binary-tree-encoding": [], "standard-classifier": []}
    prediction_time_dict = {'meta_binary_tree_classifier': [], "random": [], "huffman": [], "balanced-tree": [],
                            "meta-binary-tree-encoding": [], "standard-classifier": []}
    number_of_users = [i for i in range(min_users, max_users, step)]
    logger.info("number_of_users = " + str(number_of_users))
    for n_users in number_of_users:
        logger.info("DATA FOR %d CLASSES" % n_users)
        rf = RandomForest(seed=h2o_seed, ntrees=cfg.getint('random_forest', 'ntrees'),
                          max_depth=cfg.getint('random_forest', 'max_depth'),
                          categorical_encoding=cfg.get('random_forest', 'categorical_encoding'),
                          nbins_cats=cfg.getint('random_forest', 'nbins_cats'),
                          histogram_type=cfg.get('random_forest', 'histogram_type'))

        logger.info("Meta-Binary Tree Classifier")
        cls = MagicTreeClassifier(estimator=rf)
        micro_acc, macro_acc, train_time, prediction_time = prediction_results(train_df, test_df, cls, n_users,
                                                                               unique_users, n_folds,
                                                                               columns_to_keep)
        micro_accuracies['meta_binary_tree_classifier'].append(micro_acc)
        macro_accuracies['meta_binary_tree_classifier'].append(macro_acc)
        prediction_time_dict['meta_binary_tree_classifier'].append(prediction_time)
        training_time_dict['meta_binary_tree_classifier'].append(train_time)
        logger.info("*************")

        for encoding in ["random", "huffman", "balanced-tree", "meta-binary-tree-encoding"]:
            logger.info("Classification using encoding %s" % encoding)
            cls = EncodedClassifier(rf, encoding_type=encoding)
            micro_acc, macro_acc, train_time, prediction_time = prediction_results(train_df, test_df, cls, n_users,
                                                                                   unique_users, n_folds,
                                                                                   columns_to_keep)
            micro_accuracies[encoding].append(micro_acc)
            macro_accuracies[encoding].append(macro_acc)
            prediction_time_dict[encoding].append(prediction_time)
            training_time_dict[encoding].append(train_time)
            logger.info("*************")

        logger.info("Standard Classifier")
        cls = rf
        micro_acc, macro_acc, train_time, prediction_time = prediction_results(train_df, test_df, cls, n_users,
                                                                               unique_users, n_folds,
                                                                               columns_to_keep)
        micro_accuracies['standard-classifier'].append(micro_acc)
        macro_accuracies['standard-classifier'].append(macro_acc)
        prediction_time_dict['standard-classifier'].append(prediction_time)
        training_time_dict['standard-classifier'].append(train_time)
        logger.info("*************")
    logger.info("micro_accuracies = " + str(micro_accuracies))
    logger.info("macro_accuracies = " + str(macro_accuracies))
    logger.info("training_time = " + str(training_time_dict))
    logger.info("prediction_time = " + str(prediction_time_dict))

    # Plot generation
    scores_dict = {"micro accuracy": micro_accuracies, "macro accuracy": macro_accuracies,
                   "training time (seconds)": training_time_dict, "prediction time (seconds)": prediction_time_dict}
    classification_types = ["meta_binary_tree_classifier", "random", "huffman", "balanced-tree",
                            "meta-binary-tree-encoding", "standard-classifier"]
    for classification in classification_types:
        fig = plt.figure(figsize=(15, 15))
        for i, score in enumerate(scores_dict.keys()):
            ax = fig.add_subplot(2, 2, i + 1)
            ax.plot(number_of_users, scores_dict[score][classification])
            plt.title("Classification %s for %s" % (score, classification))
            plt.xlabel("Number of classes")
            plt.ylabel(score)
        plt.savefig("../images/H2O_" + classification)
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
    plt.savefig("../images/H2O_all")
    plt.close(fig)


def main():
    user_identification()
    return 0


if __name__ == '__main__':
    logger = setup_logger('user_identification')
    main()
