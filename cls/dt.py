import pydotplus
import numpy as np
# from IPython.display import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.append('..')
from data.gen_data import get_train_test

SEED = 222

'''
def print_graph(clf, feature_names):
   """ print dcision tree. """
   graph = export_graphviz(
      clf, 
      label='root',
      proportion=True,
      impurity=False, 
      out_file=None,
      feature_names=feature_names,
      class_names={0: "D", 1: "R"},
      filled=True,
      rounded=True
   )
   graph = pydotplus.graph_from_dot_data(graph)
   return graph.create_png()
   #plt.imshow(graph.create_png())
#   return Image(graph.create_png())
'''

# get the training data
xtrain, xtest, ytrain, ytest = get_train_test()

'''
# tree 1 
t1 = DecisionTreeClassifier(max_depth=1, random_state=SEED)
t1.fit(xtrain, ytrain)
p = t1.predict_proba(xtest)[:,1]
print "Decision tree ROC_AUC score: %.3f" % roc_auc_score(ytest, p)
'''

# tree 2 
t2 = DecisionTreeClassifier(max_depth=3, random_state=SEED)
t2.fit(xtrain, ytrain)
p1 = t2.predict_proba(xtest)[:,1]
#print "Decision tree ROC_AUC score: %.3f" % roc_auc_score(ytest, p)


# tree 3 
drop = ['transaction_amt']

xtrain_slim = xtrain.drop(drop, 1)
xtest_slim = xtest.drop(drop, 1)

t3 = DecisionTreeClassifier(max_depth=3, random_state=SEED)
t3.fit(xtrain_slim, ytrain)
p2 = t3.predict_proba(xtest_slim)[:, 1]
#print "Decision tree ROC_AUC score: %.3f" % roc_auc_score(ytest, p)

p = np.mean([p1, p2], axis=0)
print 'Average of decision tree ROC-AUC score: %.3f' % roc_auc_score(ytest, p)


# random forest
rf = RandomForestClassifier(
   n_estimators=15,
   max_features=5,
   random_state=SEED
)

rf.fit(xtrain, ytrain)
p = rf.predict_proba(xtest)[:,1]
print 'Average of decision tree ROC-AUC score: %.3f' % roc_auc_score(ytest, p)


# print_graph(t1, xtrain.columns)
