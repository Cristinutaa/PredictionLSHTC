import time


from sklearn.datasets import load_svmlight_file
from sklearn.naive_bayes import MultinomialNB #GaussianNB

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from numpy import shape
from sklearn.model_selection import cross_val_score

from code.statistics import statistics
from code.classifiers import EncodedClassifier


train_file = "C:/Users/Cristina/PhD Track/Data/LSHTC/LSHTC1/dry-run/train_corrected.txt"
test_file = "C:/Users/Cristina/PhD Track/Data/LSHTC/LSHTC1/dry-run/test_corrected.txt"

X_train, y_train = load_svmlight_file(train_file)
X_test, y_test = load_svmlight_file(test_file)


if shape(X_train)[1] > shape(X_test)[1]:
    X_test, y_test = load_svmlight_file(test_file, n_features=shape(X_train)[1])
else:
    X_train, y_train = load_svmlight_file(train_file, n_features=shape(X_test)[1])

y_train = [str(int(i)) for i in y_train]
y_test = [str(int(i)) for i in y_test]


# class ClassificationNode(Node):
#     def __init__(self, classes, classifier=None, left_node=None, right_node=None, level=0):
#         Node.__init__(self, left_node, right_node, level)
#         self.classes = classes
#         self.classifier = classifier
#
#
# def train_classifier(X_train, y_train, X_test, y_test, estimator):
#     y_train_set = set(y_train)
#     y_1 = set(random.sample(y_train_set, len(y_train_set) // 2))
#     y_2 = y_train_set - y_1
#     y_train_1 = [1 if y in y_1 else 0 for y in y_train]
#     y_test_1 = [1 if y in y_1 else 0 for y in y_test]
#     current_estimator = clone(estimator)
#     current_estimator.fit(X_train, y_train_1)
#     pred = current_estimator.predict(X_test)
#     acc = accuracy_score(pred, y_test_1)
#     count = 1
#     max_acc = acc
#     best_sets = [y_1, y_2]
#     best_est = clone(current_estimator)
#     while acc < 0.8 and count < 10:
#         count += 1
#         class_pred = {}
#         for j, i in enumerate(y_test):
#             if i in class_pred:
#                 class_pred[i][pred[j]] += 1
#             else:
#                 class_pred[i] = {}
#                 class_pred[i][pred[j]] = 1
#                 class_pred[i][int(not pred[j])] = 0
#         y_1 = set()
#         y_2 = set()
#         for i in class_pred:
#             if class_pred[i][1] > class_pred[i][0]:
#                 y_1.add(i)
#             else:
#                 y_2.add(i)
#         if not y_1:
#             y_1 = {y_2.pop()}
#         elif not y_2:
#             y_2 = {y_1.pop()}
#         y_train_1 = [1 if y in y_1 else 0 for y in y_train]
#         y_test_1 = [1 if y in y_1 else 0 for y in y_test]
#         current_estimator.fit(X_train, y_train_1)
#         pred = current_estimator.predict(X_test)
#         acc = accuracy_score(pred, y_test_1)
#         if acc > max_acc:
#             max_acc = acc
#             best_sets = [y_1, y_2]
#             best_est = clone(current_estimator)
#     #    if max_acc < 0.8:
#     #        print("The best acc that could be obtained in 10 iterations is %2.2f " %max_acc)
#     return best_est, best_sets[0], best_sets[1]
#
#
# def node_train(node, X_train, y_train, estimator):
#     row_dict_train = {}
#     for i, c in enumerate(y_train):
#         if c in row_dict_train:
#             row_dict_train[c].append(X_train[i])
#         else:
#             row_dict_train[c] = [X_train[i]]
#     '''
#     row_dict_test = {}
#     for i, c in enumerate(y_test):
#         if c in row_dict_test:
#             row_dict_test[c].append(X_test[i])
#         else:
#             row_dict_test[c] = [X_test[i]]
#     '''
#     if len(node.classes) > 1:
#         classifier, y_1, y_2 = train_classifier(X_train, y_train, X_train, y_train, estimator)
#         node.classifier = classifier
#         y_1_train = [y for y in y_train if y in y_1]
#         y_2_train = [y for y in y_train if y in y_2]
#         # y_1_test = [y for y in y_test if y in y_1]
#         # y_2_test = [y for y in y_test if y in y_2]
#         X_1_train = sparse.vstack([row_dict_train[i].pop(0) for i in y_1_train])
#         X_2_train = sparse.vstack([row_dict_train[i].pop(0) for i in y_2_train])
#         # X_1_test = sparse.vstack([row_dict_test[i].pop(0) for i in y_1_test])
#         # X_2_test = sparse.vstack([row_dict_test[i].pop(0) for i in y_2_test])
#         node.left_node = ClassificationNode(y_1)
#         node_train(node.left_node, X_1_train, y_1_train, estimator)
#         node.right_node = ClassificationNode(y_2)
#         node_train(node.right_node, X_2_train, y_2_train, estimator)
#         node.change_level(max(node.right_node.level, node.left_node.level) + 1)
#     else:
#         node.change_level(0)
#
#
# def create_magic_tree_encoding(X_train, y_train, estimator):
#     root = ClassificationNode(set(y_train))
#     node_train(root, X_train, y_train, estimator)
#     class_to_codes, codes_to_class = encode_classes(root, "", root.level, "classes", True)
#     return class_to_codes, codes_to_class, root.level
#
#
# def fit_single(estimator, X, y):
#     estimator = clone(estimator)
#     estimator.fit(X, y)
#     return estimator
#
#
# class EncodedClassifier(BaseEstimator, ClassifierMixin):
#
#     def __init__(self, estimator, n_jobs=1, encoding_type="random"):
#         self.estimator = estimator
#         self.n_jobs = n_jobs
#         self.encoding_type = encoding_type
#         self.estimators_ = None
#
#     # y is has Series type
#     def __encode(self, X, y):
#         if self.encoding_type == "huffman":
#             y = pd.Series(y_train)
#             self.dict_user_code, self.dict_code_user, self.encoding_length = create_encoding_huffman(y)
#         if self.encoding_type == "balanced-tree":
#             y = pd.Series(y_train)
#             self.dict_user_code, self.dict_code_user, self.encoding_length = create_balanced_tree_encoding(y)
#         if self.encoding_type == "random":
#             self.dict_user_code, self.dict_code_user, self.encoding_length = create_encoding_random(set(y))
#         if self.encoding_type == "magic-tree":
#             self.dict_user_code, self.dict_code_user, self.encoding_length = create_magic_tree_encoding(X, y,
#                                                                                                         self.estimator)
#         encoded_train = encode_users(y, self.dict_user_code, self.encoding_length)
#         return encoded_train
#
#     def fit(self, x, y=None):
#         y = self.__encode(x, y)
#         self.estimators_ = Parallel(n_jobs=self.n_jobs)(
#             delayed(fit_single)(self.estimator, x, y[column]) for _, column in enumerate(y))
#         return self
#
#     def predict(self, x):
#         results = np.array([estimator.predict(x) for estimator in self.estimators_]).T
#         results = pd.DataFrame(results)
#         y_pred = decode_users(results, self.dict_code_user)
#         return np.array(y_pred)


for encoding_type in ["magic-tree"]:
    print(encoding_type)
    cls = EncodedClassifier(SVC(class_weight="balanced", random_state=1, ), encoding_type=encoding_type)
    cls.fit(X_train, y_train)
    y_pred = cls.predict(X_test)
    statistics(y_train, y_pred, y_test, show_hist=False)
