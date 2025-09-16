Multiple Linear Regression
Multiple Linear Regression predicts a value using two or more input features (variables).
The model tries to fit a line (or hyperplane) to the data.

Model Equation
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ
Where:

    ŷ is the predicted value.
    β₀ is the intercept.
    β₁, β₂, ..., βₚ are the coefficients for each feature.
    x₁, x₂, ..., xₚ are the input features.

# Cost Function (Mean Squared Error)
The cost function measures how well the model fits the data:
    J(β) = (1/n) * Σ [yᵢ - (β₀ + β₁x₁ᵢ + β₂x₂ᵢ + ... + βₚxₚᵢ)]²
    
Where:

    J(β) is the cost function.
    yᵢ is the actual value for the i-th data point.
    x₁ᵢ, x₂ᵢ, ..., xₚᵢ are the feature values for the i-th data point.
    n is the number of data points.
    Gradient Descent
    Gradient descent updates each coefficient to minimize the cost:

# Update Rule for Each Coefficient
 For each coefficient βⱼ (where j = 0, 1, ..., p):

βⱼ := βⱼ - α * (∂J/∂βⱼ)
    α is the learning rate.
    ∂J/∂βⱼ is the partial derivative of the cost function with respect to βⱼ.

# Intuition
    The model finds the best combination of coefficients to fit the data.
    The cost function tells us how good the fit is.
    Gradient descent helps us adjust the coefficients to minimize the cost.