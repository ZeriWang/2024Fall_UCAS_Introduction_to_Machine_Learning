'''
问题背景：
    已知一组房源的信息和价格，如何根据这些信息预测新房源的价格？
训练集： [[面积]]->X_train, [价格]->y_train
    [[1661], [951], [432], [1413], [2187],[661],[1952],[856],[629],[1787],[1202]] -> X_train
    [291,288,101,309,292,144,306,229,213,337,275] -> y_train
    本问题中的训练集包含11个样本，要解决的是一个有监督问题。
测试集： [[面积]] -> X_test
    [[1501]] -> X_test
假设空间：
    线性回归模型 \ 二次回归模型 \ 三次回归模型
    linear regression / quadratic regression / cubic regression
学习到的假设模型：
    代码中“linear_model”、“quadratic_model”、"cubic_model"变量所指代的假设模型。

model output: [预测价格] -> y_pred -> ?
'''

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# prepare data
X_train = np.array([[1661], [951], [432], [1413], [2187], [661], [1952], [856], [629], [1787], [1202]]).astype(
    np.float32)  # area
y_train = np.array([291, 288, 101, 309, 292, 144, 306, 229, 213, 337, 275]).astype(np.float32)  # price
X_test = np.array([[1501]]).astype(np.float32)

# predict with linear model
## initialize linear regression model
linear_model = LinearRegression()

## train linear model
linear_model.fit(X_train, y_train)

## predict test data with trained linear model
y_pred = linear_model.predict(X_test)
print('Predicted house price with linear regression model: {:.2f}'.format(y_pred[0]))

## coefficient and intercept of trained linear model
print('coefficient a and intercept b of trained linear model y=ax+b:')
print("Coefficient a: {:.6f} \t Intercept b: {:.2f}".format(linear_model.coef_[0], linear_model.intercept_))

print('-' * 60)

# quadratic model
## create quadratic features
quadratic_poly = PolynomialFeatures(degree=2)
quadratic_X_train = quadratic_poly.fit_transform(X_train)
quadratic_X_test = quadratic_poly.fit_transform(X_test)

## initialize quadratic regression models with quadratic features
quadratic_model = LinearRegression()
quadratic_model.fit(quadratic_X_train, y_train)

## predict test data with trained quadratic model
quadratic_y_pred = quadratic_model.predict(quadratic_X_test)
print('Predicted house price with quadratic regression model: {:.2f}'.format(quadratic_y_pred[0]))

## coefficient and intercept of trained quadratic model
print(f"Model Coefficients: {quadratic_model.coef_}")
print(f"Model Intercept: {quadratic_model.intercept_}")

print('-' * 60)

# cubic model
## create cubic features
cubic_poly = PolynomialFeatures(degree=3)
cubic_X_train = cubic_poly.fit_transform(X_train)
cubic_X_test = cubic_poly.fit_transform(X_test)

## initialize cubic regression models with cubic features
cubic_model = LinearRegression()
cubic_model.fit(cubic_X_train, y_train)

## predict test data with trained cubic model
cubic_y_pred = cubic_model.predict(cubic_X_test)
print('Predicted house price with cubic regression model: {:.2f}'.format(cubic_y_pred[0]))

## coefficient and intercept of trained cubic model
print(f"Model Coefficients: {cubic_model.coef_}")
print(f"Model Intercept: {cubic_model.intercept_}")

# visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
## visualize linear regression
axes[0].scatter(X_train[:, ], y_train, color='blue', label='Actual Prices')
axes[0].scatter(X_test[:, ], y_pred, marker='*', color='fuchsia', label='Predicted Price')
axes[0].plot(np.sort(X_train[:, ]), linear_model.predict(X_train), color='red', label='Prediction Curve')
axes[0].set_xlabel('Square Feet')
axes[0].set_ylabel('Price')
axes[0].set_title('Square Feet vs Price (Linear Regression)')
axes[0].legend()

## visualize quadratic regression
axes[1].scatter(quadratic_X_train[:, 1], y_train, color='blue', label='Actual Prices')
axes[1].scatter(quadratic_X_test[:, 1], quadratic_y_pred, marker='*', color='fuchsia', label='Predicted Price')
axes[1].plot(np.sort(quadratic_X_train[:, 1]),
             quadratic_model.predict(quadratic_poly.transform(np.sort(X_train, axis=0))), color='red',
             label='Prediction Curve')
axes[1].set_xlabel('Square Feet')
axes[1].set_ylabel('Price')
axes[1].set_title('Square Feet vs Price (Quadratic Regression)')
axes[1].legend()

## visualize cubic regression
axes[2].scatter(cubic_X_train[:, 1], y_train, color='blue', label='Actual Prices')
axes[2].scatter(cubic_X_test[:, 1], cubic_y_pred, marker='*', color='fuchsia', label='Predicted Price')
axes[2].plot(np.sort(cubic_X_train[:, 1]), cubic_model.predict(cubic_poly.transform(np.sort(X_train, axis=0))),
             color='red', label='Prediction Curve')
axes[2].set_xlabel('Square Feet')
axes[2].set_ylabel('Price')
axes[2].set_title('Square Feet vs Price (Cubic Regression)')
axes[2].legend()

## display subplots
plt.tight_layout()
plt.show()
