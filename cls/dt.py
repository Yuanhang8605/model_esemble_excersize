import pydotplus
from IPython.display import Image
from sklearn.metrics import roc_auc_score

from sklearn.tree import DecisionTreeClassifier, export_graphviz

import sys
sys.path.append('/home/yuan/ml/2018-1-15-essembling_example')
from data.gen_data import get_train_test

SEED = 222

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
   return Image(graph.create_png())


# get the training data
xtrain, xtest, ytrain, ytest = get_train_test()

t1 = DecisionTreeClassifier(max_depth=1, random_state=SEED)
t1.fit(xtrain, ytrain)
p = t1.predict_proba(xtest)[:,1]

print "Decision tree ROC_AUC score: %.3f" % roc_auc_score(ytest, p)
