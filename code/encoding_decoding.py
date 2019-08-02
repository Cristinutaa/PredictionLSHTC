

from code.util import merge_dict, ComparableMixin
import pandas as pd
import numpy as np
import queue
import math
import random


# Random Encoding
def create_encoding_random(unique_classes):
    """
    :param unique_classes: container with the unique classes
    :return: a dictionary having the classes as key and binary codes as values,
             a dictionary having binary codes as key and class as value
             the length of the encoding strings
    """
    encoding_length = int(math.log(len(unique_classes), 2))
    if 2**encoding_length != len(unique_classes):
        encoding_length += 1
    dict_class_code = {}
    dict_code_class = {}
    for user in unique_classes:
        code = ''.join(random.choices(['0', '1'], k=encoding_length))
        while code in dict_code_class:
            code = ''.join(random.choices(['0', '1'], k=encoding_length))

        dict_class_code[user] = code
        dict_code_class[code] = user
    return dict_class_code, dict_code_class, encoding_length


# Basic BinaryNode Structure
class BinaryNode(ComparableMixin):
    def __init__(self, left_node=None, right_node=None, level=0):
        self.right_node = right_node
        self.left_node = left_node
        self.level = level
        self.code = ''

    def assign_code(self, code):
        self.code = code + self.level * '0'

    def change_level(self, level):
        self.level = level

    def _cmpkey(self):
        return self.value


class HuffmanBinaryNode(BinaryNode):
    def __init__(self, value, left_node=None, right_node=None, level=0):
        BinaryNode.__init__(self, left_node, right_node, level)
        self.value = value


def encode_classes(node, code, level, value_attr, unique_values):
    """
    Encode all the children nodes of the current node, builds two dictionaries that keep the encoding
    :param node: the node of the tree beginning with each to encode
    :param code: the code to be assigned to the node
    :param level: the level to be assigned to the node
    :param value_attr: the attribute of the node that designates the value
    :param unique_values: boolean value that indicates if the values are uniques for each node or not
                          If unique_values true, the value should be a container
    :return: two dictionaries:  - between values and codes (if unique_values false -> value of the dictionary - list)
                                - between codes and values
    """
    values_to_codes = {}
    codes_to_values = {}
    initial_level = node.level
    node.change_level(level)
    node.assign_code(code)
    if node.right_node:
        values_to_codes_right, codes_to_values_right = encode_classes(node.right_node, code + '1', level - 1,
                                                                      value_attr, unique_values)
    if node.left_node:
        values_to_codes_left, codes_to_values_left = encode_classes(node.left_node, code + '0', level - 1,
                                                                    value_attr, unique_values)
    if initial_level == 0:
        value = getattr(node, value_attr)
        if unique_values:
            value = value.pop()
            values_to_codes[value] = node.code
        else:
            values_to_codes[value] = [node.code]
        codes_to_values[node.code] = value
        return values_to_codes, codes_to_values
    if unique_values:
        values_to_codes = {**values_to_codes_right, **values_to_codes_left}
    else:
        values_to_codes = merge_dict(values_to_codes_left, values_to_codes_right)
    codes_to_values = {**codes_to_values_left, **codes_to_values_right}
    return values_to_codes, codes_to_values


def build_huffman_tree(list_of_values):
    """
    Builds the Huffman Tree structure (nodes, left node, right node, values)
    :param list_of_values: The list of values the tree should use
    :return: the root node of the tree
    """
    p = queue.PriorityQueue()
    for value in list_of_values:
        p.put((value, HuffmanBinaryNode(value)))
    while p.qsize() > 1:
        left_value, left_node, right_value, right_node = *p.get(), *p.get()
        left_level = left_node.level
        right_level = right_node.level
        root_value = left_value + right_value
        root_level = max(left_level, right_level) + 1
        p.put((root_value, HuffmanBinaryNode(root_value, left_node, right_node, root_level)))

    root = p.get()
    if type(root) is tuple:
        return root[1]
    else:
        return HuffmanBinaryNode(root)


def create_encoding_huffman(users):
    """
    Main function for Huffman Encoding. Builds the tree, calculates the codes for every class and returns the dicts
    :param users: the list of users to encode. The frequency of the users is going to be calculated using this list
    :return: two dictionaries: - between classes and codes
                               - between codes and classes
    """
    count_users = users.value_counts()
    freq_list = list(count_users)
    root = build_huffman_tree(freq_list)
    dict_value_code, _ = encode_classes(root, '', root.level, "value", False)
    dict_class_code = {}
    dict_code_class = {}
    for index, value in count_users.iteritems():
        code = dict_value_code[value].pop(0)
        dict_class_code[index] = code
        dict_code_class[code] = index
    return dict_class_code, dict_code_class, root.level


def create_balanced_tree_encoding(users):
    """
    Main function for Balanced Tree encoding. Calculates the codes for every class and returns the dicts
    :param users: the list of users to encode. The frequency of the users is going to be calculated using this list
    :return: two dictionaries: - between classes and codes
                               - between codes and classes
    """
    count_users = users.value_counts()
    freq_list = list(count_users)
    value_to_codes, max_len = assign_code(freq_list)
    dict_user_code = {}
    dict_code_user = {}
    for index, value in count_users.iteritems():
        code = value_to_codes[value].pop(0)
        dict_user_code[index] = code
        dict_code_user[code] = index
    return dict_user_code, dict_code_user, max_len


def encode_users(users, dict_user_code, encoding_length):
    """
    Encode the users from the list
    :param users: The list of users to be encoded
    :param dict_user_code: The dict with users as key and their binary code as value
    :param encoding_length: The length of the binary codes
    :return: a pandas Dataframe with vectors corresponding to classes.
            Column names are "col_[x]" where [x] is from 1 to encoding_length
    """
    col_names = ['col_'+str(i) for i in range(encoding_length)]
    encoded_users = pd.DataFrame(index=np.arange(len(users)), columns=col_names)
    i = 0
    for user in users:
        encoded_users.iloc[i] = list(dict_user_code[user])
        i += 1
    return encoded_users


def decode_users(encoded_users, dict_code_user):
    """
    Transform a pandas Dataframe with binary values and number of columns equal to the keys from dict_code_user
    to a list of classes
    :param encoded_users: a pandas Dataframe with binary values and
                          number of columns equal to the keys from dict_code_user
    :param dict_code_user: a dict to transform the binary string codes to users.
    :return: the list of users that have be obtained during decoding
    """
    users = ['' for _ in range(len(encoded_users))]
    i = 0
    encoded_user_list = encoded_users.to_string(header=False, index=False, index_names=False).split('\n')
    encoded_user_str_list = [''.join(elements.split()) for elements in encoded_user_list]
    for string in encoded_user_str_list:
        if string in dict_code_user:
            users[i] = dict_code_user[string]
            i += 1
        else: 
            closest_class = closest_word(string, list(dict_code_user.keys()))
            users[i] = dict_code_user[closest_class]
            i += 1
    return users


def humming_distance(word1, word2):
    """
    Calculated the bitwise distance between two words.
    If one word is longer than another, the difference of length is added to the result.
    :param word1: first word
    :param word2: second word
    :return: the number of different characters between the two words
    """
    count = abs(len(word1)-len(word2))
    for j in range(len(min(word1, word2, key=len))):
        if word2[j] != word1[j]:
            count += 1
    return count


def closest_word(word, list_word):
    """
    Calculated the closest word from the list_word to the word
    :param word: the word whose nearest neighbour should be calculated
    :param list_word: the list of words to be considered
    :return: the nearest word from list_word to the word
    """
    distance = []
    for i in list_word:
        distance.append(humming_distance(word, i))
    return list_word[distance.index(min(distance))]


def assign_code(values_list, code=''):
    """
    Method to assign the codes for the Balanced Tree Encoding
    :param values_list: list of values to encode
    :param code: the code for the current "node"
    :return: values_to_codes - dict with list of codes for every value,
             max_len - the length of the codes
    """
    values_to_codes = {}
    if len(values_list) == 1:
        values_to_codes[values_list[0]] = [code]
        return values_to_codes, len(code)
    right_side = False
    right_list = []
    left_list = []
    while len(values_list):
        value = random.choices(values_list, weights=values_list, k=1).pop()
        values_list.remove(value)
        if right_side:
            right_list.append(value)
            right_side = False
        else:
            left_list.append(value)
            right_side = True
    values_to_codes_left, max_len_left = assign_code(left_list, code+'0')
    values_to_codes_right, max_len_right = assign_code(right_list, code+'1')
    max_len = max(max_len_right, max_len_left)
    values_to_codes = merge_dict(values_to_codes_left, values_to_codes_right, max_len)
    return values_to_codes, max_len

