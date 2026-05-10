import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# 1. 数据加载
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.to_numpy().astype(np.float32) / 255.0
y = mnist.target.to_numpy().astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, random_state=42
)

# 2. 层定义
class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2. / in_features)
        self.b = np.zeros((1, out_features))

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, grad_output):
        self.dW = np.dot(self.x.T, grad_output)
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        return np.dot(grad_output, self.W.T)


class ReLU:
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, grad_output):
        return grad_output * self.mask

# 3. Softmax + Loss
def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)


def cross_entropy(pred, y):
    m = y.shape[0]
    loss = -np.log(pred[np.arange(m), y] + 1e-9)
    return np.mean(loss)


def grad_softmax_crossentropy(pred, y):
    m = y.shape[0]
    grad = pred.copy()
    grad[np.arange(m), y] -= 1      
    return grad / m


# 4. 网络
class NeuralNetwork:
    def __init__(self):
        self.fc1 = Linear(784, 128)
        self.relu1 = ReLU()
        self.fc2 = Linear(128, 64)
        self.relu2 = ReLU()
        self.fc3 = Linear(64, 10)

    def forward(self, x):
        x = self.fc1.forward(x)
        x = self.relu1.forward(x)
        x = self.fc2.forward(x)
        x = self.relu2.forward(x)
        x = self.fc3.forward(x)
        return softmax(x)

    def backward(self, grad):
        grad = self.fc3.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.fc2.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.fc1.backward(grad)


# 5. 参数更新
def update(model, lr):
    for layer in [model.fc1, model.fc2, model.fc3]:
        layer.W -= lr * layer.dW
        layer.b -= lr * layer.db


# 6. 训练
def train(model, X, y, epochs=15, lr=0.01, batch_size=64):
    for epoch in range(epochs):
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]

        total_loss = 0

        for i in range(0, len(X), batch_size):
            Xb = X[i:i+batch_size]
            yb = y[i:i+batch_size]

            pred = model.forward(Xb)
            loss = cross_entropy(pred, yb)
            total_loss += loss

            grad = grad_softmax_crossentropy(pred, yb)
            model.backward(grad)

            update(model, lr)

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


# 7. 测试
def evaluate(model, X, y):
    pred = model.forward(X)
    pred_label = np.argmax(pred, axis=1)
    acc = np.mean(pred_label == y)
    print("Test Accuracy:", acc)


# 8. 运行
model = NeuralNetwork()

train(model, X_train, y_train, epochs=15, lr=0.01)
evaluate(model, X_test, y_test)