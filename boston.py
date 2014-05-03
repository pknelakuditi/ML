import numpy as np
from sklearn.svm import SVR
from matplotlib import pyplot as plt
from sklearn.datasets import  load_boston
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

# data = load_diabetes()
data = load_boston()
X = data.data
y = data.target

# prepare the training and testing data for the model
nCases = len(y)
nTrain = np.floor(nCases / 2)
trainX = X[:nTrain]
trainY = y[:nTrain]
testX  = X[nTrain:]
testY = y[nTrain:]

svr = SVR(kernel='linear', C=1.0, epsilon=0.2)
r= SVR(kernel='rbf',  C=1.0, epsilon=0.2, gamma=.0001)
log = LinearRegression()

# train both models
svr.fit(trainX, trainY)
log.fit(trainX, trainY)
r.fit(trainX, trainY)

# predict test labels from both models
predLog = log.predict(testX)
predSvr = svr.predict(testX)
predr=r.predict(testX)
# show it on the plot
plt.plot(testY, testY, label='true data')
plt.plot(testY, predSvr, 'co', label='SVR')
plt.plot(testY, predLog, 'mo', label='LogReg')
plt.plot(testY, predr, 'ro', label='SVR kernel kbf')

plt.legend()
plt.show()
