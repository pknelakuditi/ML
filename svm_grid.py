from sklearn import svm, grid_search, datasets
digits = datasets.load_digits()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
#print clf
clf.fit(digits.data, digits.target)
print clf.grid_scores_
print "---------------"
print clf.best_estimator_
print clf.best_params_
print clf.score(digits.data, digits.target)
