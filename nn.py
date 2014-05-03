from __future__ import print_function

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
#from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import neighbors

print(__doc__)
"""hi"""
# Loading the Digits dataset
digits = datasets.load_boston()

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
#n_samples = len(digits.images)
X = digits.data
y = digits.target

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'n_neighbors' : [5, 10, 15, 20], 'weights' : ['uniform', 'distance']}]



clf = GridSearchCV(neighbors.KNeighborsClassifier(n_neighbors=5), tuned_parameters, cv=3, n_jobs=1 ).fit(X_train, y_train)



print(clf.best_estimator_)

y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()
