import numpy as np

class LinearRegression1D:
    def __init__(self):
        self.params = np.array([0.1, 0.4])
    
    def predict(self, X):
        a = np.ones(X.shape)
        X1 = np.vstack([a,X])
        y_hat = np.matmul(self.params, X1)
        return y_hat
    
    def fit(self, X, y, learning_rate=0.01):
        # X.shape = (2,m), y.shape=(m,), params.shape=(2,)
        a = np.ones(X.shape)
        X1 = np.vstack([a,X])
        m = y.shape[0]
        for i in range(100):
            y_hat = self.predict(X)
            temp0 = self.params[0] - learning_rate*(1/m)*np.sum(y_hat-y)
            temp1 = self.params[1] - learning_rate*(1/m)*np.sum(np.multiply((y_hat-y), X1[1][:]))
            self.params[0] = temp0
            self.params[1] = temp1
    
if __name__ == "__main__":
    X = np.array([1,2,3,4])
    y = np.array([3,5,7,9])
    reg = LinearRegression1D()
    reg.fit(X, y)
    print(reg.predict(X))
