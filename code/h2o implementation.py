from os import path

import h2o
import numpy as np
import pandas as pd
from h2o import H2OFrame
from h2o.estimators import H2ORandomForestEstimator as RandomForest
from sklearn.metrics import accuracy_score
from code.util import get_config, timestamp, ensure_dir, setup_logger, mean_confidence_interval
from code.classifiers import create_magic_tree_encoding
from code.encoding_decoding import create_balanced_tree_encoding, create_encoding_huffman, create_encoding_random, \
    encode_users, decode_users
from code.statistics import statistics


# possible H2O column types are:
# ”unknown” - this will force the column to be parsed as all NA
# ”uuid” - the values in the column must be true UUID or will be parsed as NA
# ”string” - force the column to be parsed as a string
# ”numeric” - force the column to be parsed as numeric.
#             H2O will handle the compression of the numeric data in the optimal manner.
# ”enum” - force the column to be parsed as a categorical column.
# ”time” - force the column to be parsed as a time column. H2O will attempt to parse the following list of date time
#          formats: (date) “yyyy-MM-dd”, “yyyy MM dd”, “dd-MMM-yy”, “dd MMM yy”, (time) “HH:mm:ss”, “HH:mm:ss:SSS”,
#          “HH:mm:ss:SSSnnnnnn”, “HH.mm.ss” “HH.mm.ss.SSS”, “HH.mm.ss.SSSnnnnnn”. Times can also contain “AM” or “PM”.
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


def user_identification(n_folds, n_users, algorithm):
    cfg = get_config('user_identification.cfg')
    h2o_cfg = get_config('h2o.cfg')

    h2o.init(nthreads=h2o_cfg.getint('h2o', 'nthreads'), max_mem_size=h2o_cfg.get('h2o', 'max_mem'))
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

    # n_folds = cfg.getint('data', 'n_folds')
    # n_users = cfg.getint('data', 'n_users')

    train_frame = cfg.get('data', 'train')
    test_frame = cfg.get('data', 'test')

    if cfg.has_option('misc', 'n_jobs'):
        n_jobs = cfg.getint('misc', 'n_jobs')
    else:
        n_jobs = 1

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

    #    logger.info('Feature types:')
    #   for c in column_types:
    #        if c in ignored_columns:
    #            if c in ignored_columns_reason:
    #                logger.info('%s %s %s' % (c, column_types[c], '[ignored: %s]' % ignored_columns_reason[c]))
    #            else:
    #                logger.info('%s %s %s' % (c, column_types[c], '[ignored]'))
    #        else:
    #           logger.info('%s %s' % (c, column_types[c]))

    accuracies = []

    for fold_idx in range(n_folds):
        logger.info('fold %d/%d' % (fold_idx + 1, n_folds))

        fold_users = np.random.choice(unique_users, n_users, replace=False)
        fold_train_df = train_df.loc[train_df['user'].isin(fold_users)].reset_index(drop=True)
        fold_test_df = test_df.loc[test_df['user'].isin(fold_users)].reset_index(drop=True)
        y_train = fold_test_df['user']
        y_test = fold_test_df['user']
        columns_to_keep = [i for i in list(fold_train_df.columns) if i not in ignored_columns]
        column_types = get_h2o_column_types(list(columns_to_keep))
        fold_train_df = fold_test_df[columns_to_keep]
        fold_test_df = fold_test_df[columns_to_keep]
        if algorithm == 'huffman':
            dict_user_code, dict_code_user, encoding_length = create_encoding_huffman(fold_train_df['user'])
        elif algorithm == 'balanced_tree':
            dict_user_code, dict_code_user, encoding_length = create_balanced_tree_encoding(fold_train_df['user'])
        elif algorithm == 'random':
            dict_user_code, dict_code_user, encoding_length = create_encoding_random(fold_users)
        elif algorithm == 'magic_tree':
            cls = RandomForest(seed=h2o_seed, ntrees=cfg.getint('random_forest', 'ntrees'),
                               max_depth=cfg.getint('random_forest', 'max_depth'),
                               categorical_encoding=cfg.get('random_forest', 'categorical_encoding'),
                               nbins_cats=cfg.getint('random_forest', 'nbins_cats'),
                               histogram_type=cfg.get('random_forest', 'histogram_type'))
            dict_user_code, dict_code_user, encoding_length = create_magic_tree_encoding(fold_train_df, y_train, cls)

        encoded_train = encode_users(y_train, dict_user_code, encoding_length)
        fold_train_df = pd.concat([fold_train_df, encoded_train], axis=1)
        class_columns = encoded_train.columns
        fold_train_frame = H2OFrame(fold_train_df, column_types=column_types)
        fold_test_frame = H2OFrame(fold_test_df, column_types=column_types)

        results = pd.DataFrame(index=range(len(fold_test_frame)))
        for i in encoded_train.columns:
            # build shared RF user identification model
            fold_train_frame[i] = fold_train_frame[i].asfactor()
            rf = RandomForest(seed=h2o_seed, ntrees=cfg.getint('random_forest', 'ntrees'),
                              max_depth=cfg.getint('random_forest', 'max_depth'),
                              categorical_encoding=cfg.get('random_forest', 'categorical_encoding'),
                              nbins_cats=cfg.getint('random_forest', 'nbins_cats'),
                              histogram_type=cfg.get('random_forest', 'histogram_type'))
            rf.train(ignored_columns=list(class_columns), y=i, training_frame=fold_train_frame)
            predictions = rf.predict(fold_test_frame)

            results[i] = predictions['predict'].as_data_frame().values
        y_true = y_test.values
        y_pred = decode_users(results, dict_code_user)
        acc = accuracy_score(y_true, y_pred)
        logger.info('accuracy = %.5f' % acc)

        accuracies.append(acc)

    # statistics(y_train=y_train, y_pred=y_pred, y_test=y_true, show_hist=True,
    #           file_name="class_repartition")
    mean_acc = np.mean(np.array(accuracies))
    lower, upper = mean_confidence_interval(np.array(accuracies))
    logger.info('mean accuracy = %.5f +- %.5f' % (mean_acc, 0.5 * (upper - lower)))


def main():
    user_identification(5, 961, 'magic_tree')
    return 0


if __name__ == '__main__':
    logger = setup_logger('user_identification')
    main()
