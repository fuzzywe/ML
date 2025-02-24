Here are some **interview questions on linear regression**, ranging from basic to advanced levels. These questions cover theoretical concepts, practical applications, and coding implementation.

---

### **Basic Questions**
1. **What is linear regression?**
   - Linear regression is a statistical method used to model the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a linear equation to the observed data.

2. **What is the equation for simple linear regression?**
   - \( y = mx + b \)
     - \( y \): Dependent variable (target).
     - \( x \): Independent variable (feature).
     - \( m \): Slope (coefficient).
     - \( b \): Intercept.

3. **What is the equation for multiple linear regression?**
   - \( y = b_0 + b_1x_1 + b_2x_2 + \dots + b_nx_n \)
     - \( y \): Dependent variable.
     - \( x_1, x_2, \dots, x_n \): Independent variables.
     - \( b_0 \): Intercept.
     - \( b_1, b_2, \dots, b_n \): Coefficients.

4. **What are the assumptions of linear regression?**
   - Linearity: The relationship between features and target is linear.
   - Independence: Observations are independent of each other.
   - Homoscedasticity: The residuals have constant variance.
   - Normality: The residuals are normally distributed.
   - No multicollinearity: Features are not highly correlated with each other.

5. **What is the cost function in linear regression?**
   - The cost function is **Mean Squared Error (MSE)**:
     \[
     MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
     \]
     - \( y_i \): Actual value.
     - \( \hat{y}_i \): Predicted value.
     - \( n \): Number of observations.

6. **What is the purpose of the intercept (\( b_0 \)) in linear regression?**
   - The intercept represents the value of the dependent variable when all independent variables are zero.

7. **How do you interpret the coefficients in linear regression?**
   - A coefficient represents the change in the dependent variable for a one-unit change in the corresponding independent variable, holding all other variables constant.

8. **What is the difference between simple and multiple linear regression?**
   - Simple linear regression has one independent variable, while multiple linear regression has two or more independent variables.

---

### **Intermediate Questions**
9. **What is the difference between correlation and regression?**
   - Correlation measures the strength and direction of the relationship between two variables.
   - Regression predicts the value of a dependent variable based on one or more independent variables.

10. **What is R-squared in linear regression?**
    - R-squared (\( R^2 \)) measures the proportion of variance in the dependent variable that is explained by the independent variables. It ranges from 0 to 1, where 1 indicates a perfect fit.

11. **What is adjusted R-squared?**
    - Adjusted R-squared adjusts for the number of independent variables in the model. It penalizes adding unnecessary features that don’t improve the model.

12. **What is multicollinearity, and how does it affect linear regression?**
    - Multicollinearity occurs when independent variables are highly correlated. It can make the model unstable and inflate the variance of coefficient estimates.

13. **How do you handle multicollinearity?**
    - Remove correlated features.
    - Use dimensionality reduction techniques like PCA.
    - Apply regularization (e.g., Ridge Regression).

14. **What is the difference between Ridge and Lasso Regression?**
    - Ridge Regression adds L2 regularization to reduce the magnitude of coefficients.
    - Lasso Regression adds L1 regularization to shrink some coefficients to zero, performing feature selection.

15. **What is the role of gradient descent in linear regression?**
    - Gradient descent is an optimization algorithm used to minimize the cost function (e.g., MSE) by iteratively updating the coefficients.

16. **What are residuals in linear regression?**
    - Residuals are the differences between the actual and predicted values (\( y_i - \hat{y}_i \)).

17. **How do you check for homoscedasticity?**
    - Plot residuals vs. predicted values. If the residuals are evenly spread around zero, homoscedasticity is satisfied.

18. **What is the difference between overfitting and underfitting in linear regression?**
    - Overfitting: The model performs well on training data but poorly on unseen data (high variance).
    - Underfitting: The model performs poorly on both training and test data (high bias).

---

### **Advanced Questions**
19. **What is the closed-form solution for linear regression?**
    - The closed-form solution is given by the **normal equation**:
      \[
      \beta = (X^T X)^{-1} X^T y
      \]
      - \( \beta \): Coefficient vector.
      - \( X \): Feature matrix.
      - \( y \): Target vector.

20. **What is the difference between ordinary least squares (OLS) and gradient descent?**
    - OLS is a closed-form solution that directly computes the coefficients.
    - Gradient descent is an iterative optimization algorithm used for large datasets or when the closed-form solution is computationally expensive.

21. **How do you handle outliers in linear regression?**
    - Use robust regression techniques (e.g., RANSAC).
    - Transform the data (e.g., log transformation).
    - Remove or cap outliers.

22. **What is heteroscedasticity, and how do you detect it?**
    - Heteroscedasticity occurs when the residuals have non-constant variance.
    - Detect it using residual plots or statistical tests like the Breusch-Pagan test.

23. **What is the difference between linear regression and polynomial regression?**
    - Linear regression fits a straight line, while polynomial regression fits a curve by adding polynomial terms (e.g., \( x^2, x^3 \)).

24. **What is the role of regularization in linear regression?**
    - Regularization prevents overfitting by adding a penalty term to the cost function. Examples include Ridge (L2) and Lasso (L1) regression.

25. **How do you perform feature selection in linear regression?**
    - Use Lasso Regression to shrink coefficients to zero.
    - Use statistical tests (e.g., p-values) to select significant features.
    - Use forward/backward stepwise selection.

26. **What is the difference between R-squared and adjusted R-squared?**
    - R-squared increases as more features are added, even if they are irrelevant.
    - Adjusted R-squared penalizes adding unnecessary features.

27. **What is the impact of scaling features in linear regression?**
    - Scaling features (e.g., standardization) ensures that all features contribute equally to the model, especially when using regularization.

---

### **Coding Questions**
28. **How do you implement linear regression in Python using `scikit-learn`?**
    ```python
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    # Load dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate performance
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    ```

29. **How do you perform Ridge Regression in Python?**
    ```python
    from sklearn.linear_model import Ridge

    # Create and train the model
    model = Ridge(alpha=1.0)  # alpha is the regularization strength
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    ```

30. **How do you check for multicollinearity in Python?**
    ```python
    import pandas as pd
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif_data)
    ```

31. **How do you visualize the residuals in Python?**
    ```python
    import matplotlib.pyplot as plt

    # Calculate residuals
    residuals = y_test - y_pred

    # Plot residuals
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()
    ```

---

These questions cover a wide range of topics related to linear regression and should help you prepare for interviews! Let me know if you need further clarification or examples.
