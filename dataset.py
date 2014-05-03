from sklearn import datasets
print dir(datasets)

digits = datasets.load_boston()

print help(digits.data)
n_samples, n_features = digits.data.shape

print n_samples,n_features
print digits.data.shape
print digits.data[0]
#print digits.target_names
