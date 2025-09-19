# Cost Function in Linear Regression
    The cost function measures how well our linear regression model predicts the actual data. It calculates the difference between the predicted values and the actual values, and helps us find the best line that fits the data.

# Why is it important?
    It tells us how far off our predictions are from the real values.
    By minimizing the cost function, we find the best parameters (β₀ and β₁) for our line.
    Formula
    For a dataset with n points, the cost function (also called Mean Squared Error, MSE) is:

# J(β₀, β₁) = (1/n) * Σ [yᵢ - (β₀ + β₁xᵢ)]²
Where:

J(β₀, β₁) is the cost function.
yᵢ is the actual value for the i-th data point.
xᵢ is the input value for the i-th data point.
β₀ is the intercept (where the line crosses the y-axis).
β₁ is the slope (how steep the line is).
Σ means "sum over all data points".
# Intuition
    Lower cost means our line fits the data better.
    Higher cost means our predictions are far from the actual values.
    You can use this cost function to train your model and find the best values for β₀ and β


