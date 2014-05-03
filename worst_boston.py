import matplotlib.pyplot as plt
import numpy
from numpy import argmax
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
#from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.metrics import classification_report
from sklearn import neighbors
from sklearn.datasets.mldata import fetch_mldata
import tempfile
test_data_home = tempfile.mkdtemp()



data = fetch_mldata('uci-20070111 wine', data_home=test_data_home)
#print data.DESCR
n_trials = 3
train_percentage = [90,70,50]
# Set the parameters by cross-validation
tuned_parameters = [{'n_neighbors' : [2,3,], 'weights' : ['uniform', 'distance']}]
#test_accuracies = numpy.zeros(n_trials)
print (data.data.shape)
print (data.target.shape)
print (data.data[0])
for n in range(n_trials):
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size=train_percentage/100.0, random_state=n)
    model = GridSearchCV(neighbors.KNeighborsClassifier(n_neighbors=2), tuned_parameters, cv=2)
    model.fit(X_train, y_train)
    print model.best_estimator_
    #test_accuracies[n] = model.score(X_test, y_test)
    y_true, y_pred = y_test, model.predict(X_test)
    print(classification_report(y_true, y_pred))

#print 'Average accuracy is %f' % (test_accuracies.mean())
