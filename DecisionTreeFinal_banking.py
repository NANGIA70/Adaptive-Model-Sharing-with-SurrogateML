#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[ ]:


import zipfile
import io

# In[ ]:


import pandas as pd
from sklearn import preprocessing
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# ## Decision Tree

# In[ ]:


class TreeNode:
  def __init__(self, best_split, X, y, left_node = None, right_node = None, left_prob = 0, right_prob = 0, isLeaf = False):

    # This variable contains the prediction in case this node is the leaf
    self.best_split = best_split
    self.left_node = left_node
    self.right_node = right_node
    self.left_prob = left_prob
    self.right_prob = right_prob
    self.X = X
    self.y = y
    self.isLeaf = isLeaf


class DecisionTree:
    def __init__(self, max_depth=None, randomWalkIter = 100):
        self.max_depth = max_depth
        self.randomWalkIter = randomWalkIter


    def calculate_feature_importance(self):
      n = self.randomWalkIter
      self.importance_dict = {}
      for i in self.train_features:
        self.importance_dict[i] = 0

      for _ in range(n):
        self.run_random_walk()

      for i in self.importance_dict.keys():
        self.importance_dict[i] /= n
      # print(n)

    def run_random_walk(self):
      steps = 0
      temp = copy.deepcopy(self.importance_dict)
      for i in temp.keys():
        temp[i] = 0
      tree = copy.deepcopy(self.treeNode)
      while True:
        if tree.isLeaf:
          break
        temp[self.train_features[tree.best_split[0]]] += 1
        steps += 1
        if tree.left_node != None and tree.right_node != None:
          tree = random.choices([tree.left_node, tree.right_node], weights=[tree.left_prob, tree.right_prob])[0]
        else:
          tree = random.choice([tree.left_node, tree.right_node])
      for i in self.importance_dict.keys():
        self.importance_dict[i] += (temp[i] / steps)

    def get_max_corr(self, X_train, y_train, feat):
      df = X_train.copy()
      df_in = X_train.copy()
      y_train = y_train.copy()
      drop_list = []
      for i in df_in.columns:
        if i not in self.prediction_feature_list and i != feat:
          drop_list.append(i)

      df = df.drop(drop_list, axis=1)
      lst = dict(abs(df.corr()[feat]))
      del lst[feat]
      df = df.drop(feat, axis=1)
      df['label'] = y_train
      label_lst = dict(abs(df.corr()['label']))
      del label_lst["label"]


      final_dict = {}
      for i in lst:
        final_dict[i] = lst[i] + label_lst[i]
      sorted_corr = sorted(final_dict.items(), key = lambda x : x[1], reverse = True)
      return sorted_corr

    def fit(self, X, y):
      self.train_features = list(X.columns)
      self.X_train = X.copy()
      self.y_train = y.copy()
      self.tree, self.treeNode = self._grow_tree(X.to_numpy(), y.to_numpy())
      self.calculate_feature_importance()

    def get_best_split(self, feature, X):
      feature_idx = self.train_features.index(feature)
      best_gini = np.inf
      best_split = None
      best_left_indices = None
      best_right_indices = None
      thresholds = np.unique(X[:, feature_idx])
      for threshold in thresholds:
          left_indices = np.where(X[:, feature_idx] <= threshold)[0]
          right_indices = np.where(X[:, feature_idx] > threshold)[0]

          if len(left_indices) == 0 or len(right_indices) == 0:
              continue

          gini = self._gini_impurity(y[left_indices], y[right_indices])

          if gini < best_gini:
              best_gini = gini
              best_split = (feature_idx, threshold)
              best_left_indices = left_indices
              best_right_indices = right_indices
      return best_split

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))

        if (self.max_depth is not None and depth >= self.max_depth) or num_classes == 1:
            final_class = int(np.bincount(y).argmax())
            return final_class, TreeNode(final_class, X = X, y = y, isLeaf=True)

        best_gini = np.inf
        best_split = None
        best_left_indices = None
        best_right_indices = None

        for feature_idx in range(num_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature_idx] <= threshold)[0]
                right_indices = np.where(X[:, feature_idx] > threshold)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                gini = self._gini_impurity(y[left_indices], y[right_indices])

                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature_idx, threshold)
                    best_left_indices = left_indices
                    best_right_indices = right_indices

        if best_gini == np.inf:
            final_class = int(np.bincount(y).argmax())
            return final_class, TreeNode(final_class, X = X, y = y, isLeaf=True)


        left_subtree, leftNode = self._grow_tree(X[best_left_indices], y[best_left_indices], depth + 1)
        right_subtree, rightNode = self._grow_tree(X[best_right_indices], y[best_right_indices], depth + 1)

        currNode = TreeNode(best_split, X = X, y = y, left_node = leftNode, right_node = rightNode, left_prob = len(X[best_left_indices]) / len(X), right_prob = len(X[best_right_indices])/ len(X))

        return (best_split, left_subtree, right_subtree), currNode

    def _gini_impurity(self, left_y, right_y):
        p_left = len(left_y) / (len(left_y) + len(right_y))
        p_right = len(right_y) / (len(left_y) + len(right_y))
        gini_left = 1 - sum((np.bincount(left_y) / len(left_y))**2)
        gini_right = 1 - sum((np.bincount(right_y) / len(right_y))**2)
        gini = p_left * gini_left + p_right * gini_right
        return gini


    def get_inp_count(self, inp, X_train):
      drop_cols = []
      X_train = X_train.reset_index(drop = True)
      for i in list(X_train.columns):
        if i not in self.prediction_feature_list:
          drop_cols.append(i)
      X_train = X_train.drop(drop_cols, axis=1)
      occ_count = 0

      for i in X_train.index:
        flag = True
        for col in list(inp.columns):
          threshold = (max(X_train[col]) - min(X_train[col])) / 20
          if abs(inp[col][0] - X_train[col][i]) > threshold:
            flag = False
            break
        if flag:
          occ_count += 1
      return occ_count

    def get_counts(self, inp, left_data, right_data, correlated_features):

      left_count, right_count = 0, 0
      iter = 0
      drop_cols = []
      left_data = left_data.reset_index(drop = True)
      right_data = right_data.reset_index(drop = True)
      for i in list(left_data.columns):
        if i not in self.prediction_feature_list:
          drop_cols.append(i)
      left_data = left_data.drop(drop_cols, axis=1)
      right_data = right_data.drop(drop_cols, axis=1)

      inp_left = tuple(np.array(inp) / (left_data.max().to_numpy() - left_data.min().to_numpy() + 1e-10))
      inp_right = tuple(np.array(inp) / (right_data.max().to_numpy()  - right_data.min().to_numpy() + 1e-10))


      left_diff = np.abs(right_data - inp) / (left_data.max().to_numpy() - left_data.min().to_numpy() + 1e-10)
      right_diff = np.abs(right_data - inp) / (right_data.max().to_numpy()  - right_data.min().to_numpy() + 1e-10)
      # print(left_diff, right_diff)
      tolerance = 0.05
      out_left = (left_diff <= tolerance).all(axis=1)
      out_right = (right_diff <= tolerance).all(axis=1)

      left_count = out_left.sum()
      right_count = out_right.sum()

      return left_count + 1, right_count + 1



    def predict(self, X):
      self.prediction_feature_list = list(X.columns)
      return np.array([self._predict_single(x, self.tree, self.treeNode, list(X.columns))[0] for x in tqdm(X.to_numpy())])

    def _predict_single(self, x, tree, treeNode, inp_cols):
        if isinstance(tree, int) or treeNode.isLeaf:
            return tree, treeNode

        feature_idx, threshold = tree[0]
        best_feature_name = self.train_features[feature_idx]
        inp_df = pd.DataFrame([list(x)], columns=self.prediction_feature_list)

        # Logic to handle missing feature
        if best_feature_name not in self.prediction_feature_list:
          X_t = pd.DataFrame(treeNode.X, columns = self.train_features)
          y_t = pd.Series(treeNode.y)
          correlated_features = self.get_max_corr(X_t, y_t, best_feature_name)
          left_weighted_sum = 0
          right_weighted_sum  = 0
          normalization_factor = 0
          for feat, _ in correlated_features[:3]:
            out = self.get_best_split(feat, treeNode.X)
            if out == None:
              continue
            _, threshold = out
            left_data = X_t[X_t[feat] <= threshold].reset_index(drop=True)
            right_data = X_t[X_t[feat] > threshold].reset_index(drop=True)
            dropped_x = list(x)
            dropped_x.pop(inp_cols.index(feat))
            left_count, right_count = self.get_counts(np.array(dropped_x), left_data.drop(feat, axis = 1), right_data.drop(feat, axis = 1), correlated_features)
            left_weighted_sum += (self.importance_dict[feat]*left_count*treeNode.left_prob)
            right_weighted_sum += (self.importance_dict[feat]*right_count*treeNode.right_prob)
            normalization_factor += self.importance_dict[feat]

          left_dec = left_weighted_sum / (normalization_factor + 1)
          right_dec = right_weighted_sum / (normalization_factor + 1)

          if left_dec >= right_dec:
            return self._predict_single(x, tree[1], treeNode.left_node, inp_cols)
          else:
            return self._predict_single(x, tree[2], treeNode.right_node, inp_cols)

        elif inp_df[best_feature_name][0] <= threshold:
            return self._predict_single(x, tree[1], treeNode.left_node, inp_cols)
        else:
            return self._predict_single(x, tree[2], treeNode.right_node, inp_cols)




# In[ ]:


def run_smaller_model(X_train, X_test, y_train, y_test):
  model = DecisionTree(max_depth=5)
  model.fit(X_train, y_train)
  preds = model.predict(X_test)
  print(classification_report(y_test, preds, digits=4))

# In[ ]:


def run_imputation_model(model, X_train, X_test, y_train, y_test, dropped_columns):
  X_test_final = X_test.copy()
  while len(dropped_columns) != 0:
    clf = DecisionTree(max_depth=5)
    y_train_temp = X_train[dropped_columns[0]]
    # print(y_train[:10])
    if min(y_train_temp) < 0:
      y_train_temp += abs(min(y_train_temp))
    X_train_temp = X_train.drop(dropped_columns, axis=1)
    # print(X_train_temp.head())
    clf.fit(X_train_temp, y_train_temp)
    X_test_final[dropped_columns[0]] = clf.predict(X_test_final)
    dropped_columns.pop(0)
  predictions = model.predict(X_test_final)
  print(classification_report(y_test, predictions, digits=4))


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
def run_imputation_sklearn(model, X_train, X_test, y_train, y_test, dropped_columns):
    X_test_final = X_test.copy()
    while len(dropped_columns) != 0:
      clf = DecisionTreeRegressor(max_depth=3)
      y_train_temp = X_train[dropped_columns[0]]
      # print(y_train[:10])
      if min(y_train_temp) < 0:
        y_train_temp += abs(min(y_train_temp))
      X_train_temp = X_train.drop(dropped_columns, axis=1)
      # print(X_train_temp.head())
      clf.fit(X_train_temp, y_train_temp)
      X_test_final[dropped_columns[0]] = clf.predict(X_test_final)
      dropped_columns.pop(0)
    predictions = model.predict(X_test_final)
    print(classification_report(y_test, predictions, digits=4))


# In[ ]:




def run_imputation_banking(model, X_train, X_test, y_train, y_test, dropped_columns):
    X_test_final = X_test.copy()
    while len(dropped_columns) != 0:
      if dropped_columns[0] == "balance":
        clf = DecisionTreeRegressor(max_depth=3)
      else:
        clf = DecisionTree(max_depth=5)
      y_train_temp = X_train[dropped_columns[0]]
      # print(y_train[:10])
      if min(y_train_temp) < 0:
        y_train_temp += abs(min(y_train_temp))
      X_train_temp = X_train.drop(dropped_columns, axis=1)
      # print(X_train_temp.head())
      clf.fit(X_train_temp, y_train_temp)
      X_test_final[dropped_columns[0]] = clf.predict(X_test_final)
      dropped_columns.pop(0)
    predictions = model.predict(X_test_final)
    print(classification_report(y_test, predictions, digits=4))

# # Predictions

# In[ ]:


def print_tree(train_features, node):
  if node.isLeaf:
    return
  else:
    print(train_features[node.best_split[0]])
    print_tree(train_features, node.left_node)
    print_tree(train_features, node.right_node)

# ## Banking

# In[ ]:


with zipfile.ZipFile("Datasets/bank+marketing.zip") as z:
    with z.open("bank.zip") as inner_bytes:
        with zipfile.ZipFile(io.BytesIO(inner_bytes.read())) as inner:
            with inner.open("bank-full.csv") as f:
                df = pd.read_csv(f, sep=";")

# In[ ]:


df.head()

# In[ ]:


le = LabelEncoder()
text_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome", "y"]
label = "y"

for col in text_cols:
  df[col] = le.fit_transform(df[col])
  df[col] += 1

# In[ ]:


df.head()

# In[ ]:


X = df.drop(label, axis=1)
y = df[label]

X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state=42)

# In[ ]:


clf = DecisionTree(max_depth=6)
clf.fit(X_train, y_train)

# In[ ]:


X_test = X_test.iloc[:1000, :]
y_test = y_test[:1000]

# In[ ]:


df.contact.unique()

# In[ ]:


print_tree(clf.train_features, clf.treeNode)

# ### Original model

# In[ ]:


predictions  = clf.predict(X_test)
print(classification_report(y_test, predictions, digits=4))

# In[ ]:


df["housing"].unique()

# ### Binary

# In[ ]:


X_test_temp = X_test.drop("housing", axis=1)
# predictions  = clf.predict(X_test_temp)
predictions  = clf.predict(X_test_temp.reset_index(drop=True))
print(classification_report(y_test, predictions, digits=4))

# In[ ]:


run_smaller_model(X_train.drop("housing", axis=1), X_test.drop("housing", axis=1), y_train, y_test)

# In[ ]:


run_imputation_model(clf, X_train, X_test.drop("housing", axis=1), y_train, y_test, ["housing"])

# ### Numeric

# In[ ]:


X_test_temp = X_test.drop("balance", axis=1)
predictions  = clf.predict(X_test_temp.reset_index(drop=True))
print(classification_report(y_test, predictions, digits=4))

# In[ ]:


run_smaller_model(X_train.drop("balance", axis=1), X_test.drop("balance", axis=1), y_train, y_test)

# In[ ]:


run_imputation_sklearn(clf, X_train, X_test.drop("balance", axis=1), y_train, y_test, ["balance"])

# ### Ordinal

# In[ ]:


X_test_temp = X_test.drop("contact", axis=1)
predictions  = clf.predict(X_test_temp.reset_index(drop=True))
print(classification_report(y_test, predictions, digits=4))

# In[ ]:


run_smaller_model(X_train.drop("contact", axis=1), X_test.drop("contact", axis=1), y_train, y_test)

# In[ ]:


run_imputation_model(clf, X_train, X_test.drop("contact", axis=1), y_train, y_test, ["contact"])

# ### Combined

# In[ ]:


X_test_temp = X_test.drop(["housing", "balance", "contact"], axis=1)
predictions  = clf.predict(X_test_temp.reset_index(drop=True))
print(classification_report(y_test, predictions, digits=4))

# In[ ]:


run_smaller_model(X_train.drop(["housing", "balance", "contact"], axis=1), X_test.drop(["housing", "balance", "contact"], axis=1), y_train, y_test)

# In[ ]:


run_imputation_banking(clf, X_train, X_test.drop(["balance", "housing", "contact"], axis=1), y_train, y_test, ["balance", "housing" , "contact"])
