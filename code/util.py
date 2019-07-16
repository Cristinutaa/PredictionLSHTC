
import configparser
import datetime
import logging
import numpy as np
import os
import sys
import time
from os import path

import scipy


def root_path():
    current_dir = path.dirname(__file__)
    relative_root_dir = path.join(path.join(current_dir, path.pardir))
    return path.abspath(relative_root_dir)


def config_path():
    return path.join(root_path(), 'config')


def log_path():
    return path.join(root_path(), 'log')


def get_config(name):
    config = configparser.ConfigParser()
    config.readfp(open(path.join(config_path(), name)))
    return config


def timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def setup_logger(name, dir=None):

    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger.addHandler(screen_handler)
    if dir is None:
        dir = log_path()
    ensure_dir(dir)
    file_handler = logging.FileHandler(path.join(dir, '%s.log' % name), mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    lower_bound, upper_bound = scipy.stats.t.interval(confidence, len(a)-1, loc=np.mean(a), scale=scipy.stats.sem(a))
    return lower_bound, upper_bound


def check_equal(lst: list) -> bool:
    """
    Checks if a list is formed of only one value (all the elements are equal)
    :param lst: the list to check
    :return: a boolean value that indicates if the elements of the list are all equal
    """
    return lst[1:] == lst[:-1]


def adjust_len(lst: list, new_len: int = 0) -> list:
    """
    Adds a '0' at the end of all the strings from lst
    :param new_len: the new length of all the elements. It should be superior to the current maximum length
    :param lst: The list to have to strings adapted
    :return: The new list created
    """
    if not new_len:
        new_len = max(lst, key=len) + 1
    for i, j in enumerate(lst):
        lst[i] = j + '0' * (new_len - len(j))
    return lst


def merge_dict(dict_a: dict, dict_b: dict, check_len: int = 0) -> dict:
    """
    Merge two dictionaries where the values are list
    :param dict_a: Dict with values list to merge
    :param dict_b: Dict with values list to merge
    :param check_len: value to set for the strings from the lists. If value 0, parameter ignored
    :return: A merged version of the dictionaries. If common keys, the lists are combined.
    """
    final_dict = {}
    # Add all the elements from b
    for key in dict_b:
        if check_len and len(min(dict_b[key], key=len)) < check_len:
            dict_b[key] = adjust_len(dict_b[key], check_len)
        final_dict[key] = dict_b[key]

    # Add all the elements from a. If an element is already in the the dictionary, append the list
    for key in dict_a:
        if check_len and len(min(dict_a[key], key=len)) < check_len:
            dict_a[key] = adjust_len(dict_a[key], check_len)
        if key in final_dict:
            final_dict[key] = final_dict[key]+dict_a[key]
        else:
            final_dict[key] = dict_a[key]
    return final_dict


# Generic class that allows comparing user-created classes using a given criterion
class ComparableMixin(object):
    def _compare(self, other, method):
        try:
            return method(self._cmpkey(), other._cmpkey())
        except (AttributeError, TypeError):
            # _cmpkey not implemented, or return different type,
            # so I can't compare with "other".
            return NotImplemented

    def __lt__(self, other):
        return self._compare(other, lambda s, o: s < o)

    def __le__(self, other):
        return self._compare(other, lambda s, o: s <= o)

    def __eq__(self, other):
        return self._compare(other, lambda s, o: s == o)

    def __ge__(self, other):
        return self._compare(other, lambda s, o: s >= o)

    def __gt__(self, other):
        return self._compare(other, lambda s, o: s > o)

    def __ne__(self, other):
        return self._compare(other, lambda s, o: s != o)
