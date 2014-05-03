from sklearn import svm, grid_search
import optdigits as data
import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
    
    data.data('tr')
    train_features = data.getData()
    train_target = data.getTgt()
    
    data.data('cv')
    cv_features = data.getData()
    cv_target = data.getTgt()
    
    
    clf = []
    cplot = []
    scoreplot = []
    parameters = {'kernel':('rbf',), 'C':[x * x * 50 for x in range(1, 4)], 'gamma':[x * x * 0.001 for x in range(0, 2)]}
    svr = svm.SVC()
    clf = grid_search.GridSearchCV(svr, parameters, verbose=5)
    clf.fit(np.vstack([train_features,cv_features]), np.concatenate([train_target, cv_target]))
