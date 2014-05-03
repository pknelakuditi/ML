
import matplotlib.pyplot as plt
import numpy
from numpy import argmax
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score

data = load_iris()
#print data.DESCR
n_trials = 3
train_percentage = [90,70,50]
# Set the parameters by cross-validation
tuned_parameters = [{'C': [10**i for i in range(-4,5)]}] 
print "All used parameters :",tuned_parameters
test_accuracies = numpy.zeros(n_trials)

for n in train_percentage:
    print "Logistic regression for iris data"
    print ""
    print ""
    
    print "training percentage ::",n
    print ""
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size=n/100.0, random_state=n)
    model = GridSearchCV(LogisticRegression(), tuned_parameters, cv=5)
    model.fit(X_train, y_train)
    print model.best_estimator_
    #test_accuracies[n] = model.score(X_test, y_test)
    y_true, y_pred = y_test, model.predict(X_test)
    print(classification_report(y_true, y_pred))
    print ("score")
    print ""
    print (model.best_params_)
    print ("best_params")
    print ""
    pred = model.predict(X_test)
    print (accuracy_score(y_test, pred))
    print ("accuracy_score")
    print ""
    

#print 'Average accuracy is %f' % (test_accuracies.mean())
