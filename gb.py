import matplotlib.pyplot as plt
import numpy
from numpy import argmax
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

data = load_digits()
#print data.DESCR
n_trials = 10
train_percentage = 90
# Set the parameters by cross-validation
tuned_parameters = [{'n_estimators': [100, 200, 500, 1000, 2000, 3000], 'min_samples_split':[1, 2, 3, 4, 5, 10, 15, 20],
                         'max_depth':[2, 3, 4] ,
                        'max_depth':[0.1, 0.01, 0.05],
                         'min_samples_leaf':[1, 5, 10]}]
test_accuracies = numpy.zeros(n_trials)

for n in range(n_trials):
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size=train_percentage/100.0, random_state=n)
    model = GridSearchCV(GradientBoostingClassifier(random_state=0, subsample=0.5), tuned_parameters)
    model.fit(X_train, y_train)
    print model.best_estimator_
    test_accuracies[n] = model.score(X_test, y_test)
    y_true, y_pred = y_test, model.predict(X_test)
    print(classification_report(y_true, y_pred))

print 'Average accuracy is %f' % (test_accuracies.mean())
