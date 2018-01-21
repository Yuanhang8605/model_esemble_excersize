from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline

SEED = 222

def get_models():
   """ Generate a library of base learners. """
   nb = GaussianNB()
   svc = SVC(C=100, probability=True)
   knn = KNeighborsClassifier(n_neighbors=3)
   lr = LogisticRegression(C=100, random_state=SEED)
   nn = MLPClassifier((80, 10), early_stopping=False, random_state=SEED)
   gb = GradientBoostingClassifier(n_estimators=100, random_state=SEED)
