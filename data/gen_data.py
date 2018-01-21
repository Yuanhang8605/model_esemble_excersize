import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Training and test set
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os

def get_train_test(test_size=0.95):
   """Split Data into train and test sets."""
   ### Import data
   # Always good to set a seed for reproducibility
   SEED = 222
   np.random.seed(SEED)
   df = pd.read_csv('../data/input.csv')

   y = 1 * (df.cand_pty_affiliation == "REP")
   X = df.drop(["cand_pty_affiliation"], axis=1)
   X = pd.get_dummies(X, sparse=True)
   X.drop(X.columns[X.std() == 0], axis=1, inplace=True)

   return train_test_split(X, y, test_size=test_size, random_state=SEED)

'''
xtrain, xtest, ytrain, ytest = get_train_test()
# A look at the data
print("\nExample data:")
# df.head()

df.cand_pty_affiliation.value_counts(normalize=True).plot(
   kind='bar', title='Share of No. donations'
)
plt.show()
'''