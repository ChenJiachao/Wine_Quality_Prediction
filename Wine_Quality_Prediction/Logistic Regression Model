import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
red = pd.read_csv('winequality-red.csv')
white = pd.read_csv('winequality-white.csv')

# Visualize the dataset, randomly choose them
red = shuffle(red, random_state = 10)
white = shuffle(white, random_state = 10)
red.head(10)
white.head(10)

from sklearn.model_selection import train_test_split
X_red = red.iloc[:, :-1]
y_red = red.iloc[:, -1] >= 6

X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(X_red, y_red, test_size=0.3, random_state = 0)

X_white = white.iloc[:, :-1]
y_white = white.iloc[:, -1] >= 6

X_train_white, X_test_white, y_train_white, y_test_white = train_test_split(X_white, y_white, 
                                                                            test_size=0.3, random_state = 0)

from sklearn.linear_model import LogisticRegression

LRR = LogisticRegression(solver = 'sag', max_iter = 10000)
LRR.fit(X_train_red,y_train_red)
test_score = LRR.score(X_test_red,y_test_red)


print('The Testing Accuracy for red wine is: ' + str(test_score) + '.')

LRW = LogisticRegression(solver = 'sag', max_iter = 10000)
LRW.fit(X_train_white,y_train_white)
test_score = LRW.score(X_test_white,y_test_white)


print('The Testing Accuracy for white wine is: ' + str(test_score) + '.')

# Used Stochastic Gradient and L2 regularization

# Red Wine
N = np.array(range(0,15))
alpha = 0.00001*(4**N)
AccuracyTrain = []
AccuracyTest = []

from sklearn.metrics import accuracy_score
for i in N:
    LRRed = LogisticRegression(penalty = 'l2',C = alpha[i], solver = 'sag', max_iter = 10000)
    LRRed.fit(X_train_red,y_train_red)
    
    AccuracyTrain.append(LRRed.score (X_train_red,y_train_red ) )
    AccuracyTest.append(LRRed.score (X_test_red,y_test_red ))

plt.figure(1)
plt.semilogx(alpha, AccuracyTrain,label = 'Test')
plt.semilogx(alpha, AccuracyTest, label = 'Train')
plt.legend()

max_index = AccuracyTest.index(max(AccuracyTest))
LRRed = LogisticRegression(penalty = 'l2',C = alpha[max_index], solver = 'sag', max_iter = 10000)
LRRed.fit(X_train_red,y_train_red)
accuracy = LRRed.score (X_test_red,y_test_red )
print(accuracy)

AccuracyTrain = []
AccuracyTest = []

# White Wine

for i in N:
    LRRed = LogisticRegression(penalty = 'l2',C = alpha[i], solver = 'sag', max_iter = 10000)
    LRRed.fit(X_train_white,y_train_white)
    
    AccuracyTrain.append(LRRed.score (X_train_white,y_train_white ) )
    AccuracyTest.append(LRRed.score (X_test_white,y_test_white ))

plt.figure(2)
plt.semilogx(alpha, AccuracyTrain,label = 'Test')
plt.semilogx(alpha, AccuracyTest, label = 'Train')
plt.legend()

max_index = AccuracyTest.index(max(AccuracyTest))
LRRed = LogisticRegression(penalty = 'l2',C = alpha[max_index], solver = 'sag', max_iter = 10000)
LRRed.fit(X_train_white,y_train_white)
accuracy = LRRed.score (X_test_white,y_test_white )
print(accuracy)





