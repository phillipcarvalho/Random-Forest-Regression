import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_SalariesA.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#training model using decision tree regression
from sklearn.ensemble import RandomForestRegressor
dt=RandomForestRegressor(n_estimators=10,random_state=0)
dt.fit(X,y)

#predicting
print(dt.predict([[6.5]]))

#visualising 
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, dt.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()