# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


bmi_life_data = pd.read_csv('bmi_and_life_expectancy.csv')
X = (bmi_life_data.iloc[:, 2].values)
y = (bmi_life_data.iloc[:, 1].values)

X = np.array(X.reshape(-1,1))
y = np.array(y.reshape(-1,1))

bmi_life_model = LinearRegression()
bmi_life_model.fit(X, y)

laos_life_exp = bmi_life_model.predict(21.07931)
print (laos_life_exp)

plt.scatter(X, y, color = 'blue')
plt.plot(X, bmi_life_model.predict(X), color = 'blue')
plt.title('life expectancy prediction')
plt.xlabel('BMI')
plt.ylabel('life expectancy')
plt.show()