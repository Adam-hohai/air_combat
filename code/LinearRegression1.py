from air_combat.code.txtToCsv import TxtToCsv
from air_combat.code.const import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import csv

#这版有问题

if __name__ == '__main__':
    lines = TxtToCsv().convert()


    # index = 0
    # pre = -2
    # first_index = []
    # for line in lines:
    #     if line[5] == '0.0':
    #         if index > pre + 1:
    #             print('断层')
    #             first_index.append(index)
    #         print(index)
    #         pre = index
    #     index = index + 1
    # print(first_index)
    # index = 0
    # for i in range(len(first_index)-1):
    #     with open('result'+str(index)+'.csv', 'w') as f:
    #         write = csv.writer(f)
    #         write.writerows(lines[first_index[index]:first_index[index+1]])
    #     index = index + 1


    x = []
    y = []
    for line in lines:
        x.append(float(line[0]))
        y.append(float(line[1]))
    c = {'x' : x, 'y' : y}
    datas = pd.DataFrame(c)
    #print(datas.head())

    y = datas.iloc[328500:328800,[1]]
    x = datas.iloc[328500:328800,[0]]
    x_train = x[:250]
    x_test = x[250:]
    y_train = y[:250]
    y_test = y[250:]

    model = LinearRegression() #线性回归
    model.fit(x_train, y_train)
    y_predicts = model.predict(x_test)  # 预测值
    R2 = model.score(y_test, y_predicts)  # 拟合程度 R2
    print('R2 = %.2f' % R2)  # 输出 R2
    coef = model.coef_  # 斜率
    intercept = model.intercept_  # 截距
    print(model.coef_, model.intercept_)  # 输出斜率和截距

    y_pre = coef * x_train + intercept
    plt.scatter(x, y, label='data')
    plt.plot(x_train, y_pre, color='green')
    plt.plot(x_test,y_predicts, color='red', label='predict')
    plt.legend()
    plt.show()

