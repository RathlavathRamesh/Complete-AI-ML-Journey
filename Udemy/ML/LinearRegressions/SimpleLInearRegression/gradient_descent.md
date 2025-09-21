# CONVERGENCE ALGORITHMS

# Gradient Descent in Linear Regression
    Gradient Descent is an optimization algorithm used to find the best parameters (β₀ and β₁) for our linear regression model by minimizing the cost function.

# Why is it important?
    It helps us find the line that best fits the data by adjusting the parameters step by step.
    It works even when we have lots of data or many parameters.

# How does it work?
    Start with random values for β₀ and β₁.
    Calculate the cost using the cost function.
    Find the gradient (the direction and rate of fastest increase of the cost).
    Update the parameters in the opposite direction of the gradient (to decrease the cost).
    Repeat steps 2–4 until the cost is as low as possible (converges).

Formula
For each parameter, the update rule is:

β₀ := β₀ - α * (∂J/∂β₀)
β₁ := β₁ - α * (∂J/∂β₁)
Where:

# := means "update to".
α is the learning rate (a small positive number that controls the step size).
∂J/∂β₀ and ∂J/∂β₁ are the partial derivatives of the cost function with respect to β₀ and β₁.

Partial Derivatives for Linear Regression
Step-by-step:
Compute the gradients for β₀ and β₁.
Update β₀ and β₁ using the formulas above.
Repeat until the cost stops decreasing.
Intuition
Gradient tells us how to change β₀ and β₁ to reduce the cost.
Learning rate controls how big each step is.
Repeat until the parameters give the lowest possible cost.
Gradient Descent is a powerful tool for training machine learning models by finding the best parameters!

Gradient Descent is a powerful tool for training machine learning models by finding the best parameters!
