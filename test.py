import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

# Extract data
data = pd.read_csv('jaybob.csv')
print(data.head())
x = data['Age'].values.reshape(-1, 1)
y = data['Odometer'].values.reshape(-1, 1)

# Build model
model = LinearRegression().fit(x, y)
X = np.linspace(np.min(x), np.max(x)).reshape(-1, 1)
yhat = model.predict(X) # predict using OLS

# Display regression plot
plt.scatter(x, y, color='blue', label='Samples')
plt.plot(X, yhat, color='red', label='Line of Regression')
plt.show()

# Get generalized statistics
x1 = sm.add_constant(x)

