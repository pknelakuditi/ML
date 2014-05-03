from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


ds = datasets.load_digits()

rf = RandomForestClassifier(n_estimators = 50)
rf.fit(ds.data,ds.target)
print rf.score(ds.data,ds.target) 


