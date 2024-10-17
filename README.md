# EX1 Implementation of Univariate Linear Regression to fit a straight line using Least Squares
## AIM:
To implement univariate Linear Regression to fit a straight line using least squares.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y.
2. Calculate the mean of the X -values and the mean of the Y -values.
3. Find the slope m of the line of best fit using the formula. 
<img width="231" alt="image" src="https://user-images.githubusercontent.com/93026020/192078527-b3b5ee3e-992f-46c4-865b-3b7ce4ac54ad.png">
4. Compute the y -intercept of the line by using the formula:
<img width="148" alt="image" src="https://user-images.githubusercontent.com/93026020/192078545-79d70b90-7e9d-4b85-9f8b-9d7548a4c5a4.png">
5. Use the slope m and the y -intercept to form the equation of the line.
6. Obtain the straight line equation Y=mX+b and plot the scatterplot.
 
## Program:
```
/*
Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by: kavipriya s.p
RegisterNumber: 2305002011
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/ex1.csv')
df.head()
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df[['X']],df['Y'],test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(x,lr.predict(x),color='red')
m=lr.coef_
m
b=lr.intercept_
b
pred=lr.predict(x_test)
pred
x_test
y_test
from sklearn.metrics import mean_squared_log_error
mse=mean_squared_log_error(y_test,pred)
mse
```

## Output:

![Screenshot 2024-10-17 094459](https://github.com/user-attachments/assets/9a746d89-a0f0-4a45-a017-46c4a50f7f30)


![Screenshot 2024-10-17 094337](https://github.com/user-attachments/assets/7448bbf9-e9c2-4428-92af-1003a44b84e0)


![Screenshot 2024-10-17 094535](https://github.com/user-attachments/assets/13de6e03-5359-4bc7-8b5d-ed5f0789f306)

![Screenshot 2024-10-17 094603](https://github.com/user-attachments/assets/d0521718-579f-4a90-a0b1-e76329ed3187)

![Screenshot 2024-10-17 094638](https://github.com/user-attachments/assets/36c70904-d39b-443a-b482-fb9a9edca971)

![Screenshot 2024-10-17 094715](https://github.com/user-attachments/assets/d68572a4-e2e3-4e87-99c8-705b4b9154f5)

![Screenshot 2024-10-17 094720](https://github.com/user-attachments/assets/3cd8fcb1-0808-411d-a63b-2224ca9de376)

![Screenshot 2024-10-17 094731](https://github.com/user-attachments/assets/b9f0a27e-e063-45f4-b19f-fc09172ac9d0)

![Screenshot 2024-10-17 094739](https://github.com/user-attachments/assets/339ae9ea-6376-4763-98ed-bc6520652705)

![Screenshot 2024-10-17 094747](https://github.com/user-attachments/assets/890ebd0c-0ddd-4ca2-909c-c33f6f818b24)


## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
