import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.losses = []

        # Gradient Descent
        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Optional: Record loss
            loss = np.mean((y_pred - y)**2)
            self.losses.append(loss)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        """Calculates the R^2 score."""
        y_pred = self.predict(X)
        u = np.sum((y - y_pred)**2) # Residual sum of squares
        v = np.sum((y - np.mean(y))**2) # Total sum of squares
        return 1 - (u / v)

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_regression
    import matplotlib.pyplot as plt

    # Generate synthetic data
    X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    mse = np.mean((predictions - y_test)**2)
    r2 = model.score(X_test, y_test)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Plotting (optional, if running locally only)
    # plt.scatter(X_test, y_test, color='black')
    # plt.plot(X_test, predictions, color='blue', linewidth=3)
    # plt.show()
