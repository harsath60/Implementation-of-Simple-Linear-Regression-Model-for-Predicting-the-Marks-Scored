# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program and import required libraries.
2. Create and load the employee dataset into a DataFrame.
3. Separate input features and target variable (Churn).
4. Split the dataset into training and testing sets.
5. Train the Decision Tree Classifier model and predict test results.
6. Evaluate accuracy, display classification report, plot the tree, and stop.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: HARSATH S
RegisterNumber: 212225230096
*/
```
```
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample data (X = input, y = output)
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Print results
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

# Plot the data and regression line
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()

/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 
RegisterNumber:  
*/
```

## Output:
<img width="909" height="573" alt="image" src="https://github.com/user-attachments/assets/e26399f5-aae5-46b6-b9b9-13bba455ae3d" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
