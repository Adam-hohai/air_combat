from air_combat.code.txtToCsv import TxtToCsv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from air_combat.code.txtToCsv import TxtToCsv
from sklearn.model_selection import train_test_split


class LinearRegression2:

    def __init__(self, first, end):
        self.first = first
        self.end = end

    def linear_regression2(self):
        lines = TxtToCsv().convert()
        current_a = []
        current_b = []
        current_c = []
        y = []
        for line in lines[self.first:self.end-1]:
            current_a.append(float(line[9]))
            current_b.append(float(line[10]))
            current_c.append(float(line[11]))
        for line in lines[self.first+1:self.end]:
            y.append(float(line[9]))
        data = {'current_a': current_a, 'current_b': current_b, 'current_c': current_c, 'y': y}
        pd_data = pd.DataFrame(data)
        pd_data.astype(float)
        #print(pd_data.columns.values.tolist())
        X = pd_data.loc[:, ['current_a', 'current_b', 'current_c']]
        y = pd_data.loc[:, ['y']]
        # X_train = X[self.first:self.end - 1]
        # X_test = X[self.end - 1]
        # y_train = y[self.first:self.end - 1]6
        # y_test = y[self.end - 1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state=100)
        # print(X.shape)
        # print(y.shape)
        # print(X_train.shape)
        # print(y_train.shape)
        # print(X_test.shape)
        # print(y_test.shape)

        model = LinearRegression()
        model.fit(X_train, y_train)
        print(model.intercept_)
        print(model.coef_)

        y_pred = model.predict(pd.DataFrame({'current_a':[float(lines[end][9])], 'current_b':[float(lines[end][10])], 'current_c':[float(lines[end][11])]}))
        y_test = float(lines[end+1][9])
        print(float(lines[end+1][9]))
        print(y_pred)
        return y_test, y_pred[0][0]


if __name__ == '__main__':
    first = 328500
    end = 0
    # y_test = []
    # y_pred = []
    test = []
    pred = []
    t = []
    # y_test, y_pred = LinearRegression2(328500, 328600).linear_regression2()
    for i in range(328600,328610,1):
        end = i
        y_test, y_pred = LinearRegression2(first, end).linear_regression2()
        test.append(y_test)
        pred.append(y_pred)
    for i in range(len(test)):
        t.append(i)
    for i in range(len(test)):
        mse = pow(pred[i]-test[i], 2)
        print(mse)
    plt.scatter(t, test, label='test', color='red', s=16.)
    plt.scatter(t, pred, label='pred', color='blue', s=16.)
    plt.legend()
    plt.show()
