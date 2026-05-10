import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# 数据加载
print("Loading MNIST...")

mnist = fetch_openml("mnist_784")
X = mnist.data.to_numpy().astype(np.float32)
y = mnist.target.to_numpy().astype(np.int32)
X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=10000,random_state=42
)

# 加噪声
def add_gaussian_noise(X, sigma=0.15):
    noise = np.random.normal(0, sigma, X.shape)
    X_noisy = X + noise
    return np.clip(X_noisy, 0, 1)


def add_random_block(X, block_size=8):
    X_new = X.copy()

    for i in range(len(X_new)):
        img = X_new[i].reshape(28, 28)

        x = np.random.randint(0, 28 - block_size)
        y = np.random.randint(0, 28 - block_size)

        img[x:x+block_size, y:y+block_size] = 0
        X_new[i] = img.reshape(-1)

    return X_new


def show_noise():
    original = X[0].reshape(28, 28)

    noisy = add_gaussian_noise(X[0:1])
    noisy = add_random_block(noisy)
    noisy = noisy[0].reshape(28, 28)

    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original")

    plt.subplot(1, 2, 2)
    plt.imshow(noisy, cmap="gray")
    plt.title("Noisy")

    plt.show()

X_train = add_gaussian_noise(X_train)
X_train = add_random_block(X_train)
X_test = add_gaussian_noise(X_test)
X_test = add_random_block(X_test)
show_noise()

# 网络基础模块
class ReLU:
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask
    def backward(self, grad):
        return grad * self.mask


class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(
            in_features,
            out_features
        ) * np.sqrt(2 / in_features)

        self.b = np.zeros(out_features)
        # Adam 优化器
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)
        self.t  = 0

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.dW = self.x.T @ grad
        self.db = np.sum(grad, axis=0)

        return grad @ self.W.T

    def adam_update(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1

        self.mW = beta1 * self.mW + (1 - beta1) * self.dW
        self.vW = beta2 * self.vW + (1 - beta2) * self.dW ** 2
        mW_hat  = self.mW / (1 - beta1 ** self.t)
        vW_hat  = self.vW / (1 - beta2 ** self.t)
        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)

        self.mb = beta1 * self.mb + (1 - beta1) * self.db
        self.vb = beta2 * self.vb + (1 - beta2) * self.db ** 2
        mb_hat  = self.mb / (1 - beta1 ** self.t)
        vb_hat  = self.vb / (1 - beta2 ** self.t)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)


class BatchNorm:
    def __init__(self, dim, eps=1e-5, momentum=0.9):
        self.gamma   = np.ones(dim)
        self.beta    = np.zeros(dim)
        self.eps     = eps
        self.momentum = momentum

        self.running_mean = np.zeros(dim)
        self.running_var  = np.ones(dim)

        # Adam states for gamma/beta
        self.mg = np.zeros(dim); self.vg = np.zeros(dim)
        self.mb = np.zeros(dim); self.vb_adam = np.zeros(dim)
        self.t  = 0

    def forward(self, x, training=True):
        if training:
            mu  = x.mean(axis=0)
            var = x.var(axis=0)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var  = self.momentum * self.running_var  + (1 - self.momentum) * var

            self.x_norm = (x - mu) / np.sqrt(var + self.eps)
            self.var    = var
            self.mu     = mu
            self.x      = x
        else:
            self.x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        return self.gamma * self.x_norm + self.beta

    def backward(self, grad):
        N = grad.shape[0]

        self.dgamma = np.sum(grad * self.x_norm, axis=0)
        self.dbeta  = np.sum(grad, axis=0)

        dx_norm = grad * self.gamma
        dvar    = np.sum(dx_norm * (self.x - self.mu) * -0.5 * (self.var + self.eps) ** -1.5, axis=0)
        dmu     = np.sum(dx_norm * -1 / np.sqrt(self.var + self.eps), axis=0) + dvar * np.mean(-2 * (self.x - self.mu), axis=0)
        dx      = dx_norm / np.sqrt(self.var + self.eps) + dvar * 2 * (self.x - self.mu) / N + dmu / N

        return dx

    def adam_update(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1

        self.mg = beta1 * self.mg + (1 - beta1) * self.dgamma
        self.vg = beta2 * self.vg + (1 - beta2) * self.dgamma ** 2
        mg_hat  = self.mg / (1 - beta1 ** self.t)
        vg_hat  = self.vg / (1 - beta2 ** self.t)
        self.gamma -= lr * mg_hat / (np.sqrt(vg_hat) + eps)

        self.mb     = beta1 * self.mb     + (1 - beta1) * self.dbeta
        self.vb_adam = beta2 * self.vb_adam + (1 - beta2) * self.dbeta ** 2
        mb_hat      = self.mb     / (1 - beta1 ** self.t)
        vb_hat      = self.vb_adam / (1 - beta2 ** self.t)
        self.beta  -= lr * mb_hat / (np.sqrt(vb_hat) + eps)


class Dropout:
    def __init__(self, p=0.3):
        self.p    = p
        self.mask = None

    def forward(self, x, training=True):
        if training:
            self.mask = (np.random.rand(*x.shape) > self.p) / (1 - self.p)
            return x * self.mask
        return x

    def backward(self, grad):
        return grad * self.mask


#  数学函数
def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(
        exp,
        axis=1,
        keepdims=True
    )

def cross_entropy(pred, y):
    m = y.shape[0]

    loss = -np.log(
        pred[np.arange(m), y] + 1e-10
    )
    return np.mean(loss)


def grad_loss(pred, y):
    m = y.shape[0]
    grad = pred.copy()
    grad[np.arange(m), y] -= 1
    return grad / m


# 网络
class NeuralNetwork:
    def __init__(self):
        self.fc1   = Linear(784, 512)
        self.bn1   = BatchNorm(512)
        self.relu1 = ReLU()
        self.drop1 = Dropout(0.3)

        self.fc2   = Linear(512, 512)
        self.bn2   = BatchNorm(512)
        self.relu2 = ReLU()
        self.drop2 = Dropout(0.3)

        self.fc3   = Linear(512, 256)
        self.bn3   = BatchNorm(256)
        self.relu3 = ReLU()
        self.drop3 = Dropout(0.2)

        self.fc4   = Linear(256, 128)
        self.bn4   = BatchNorm(128)
        self.relu4 = ReLU()

        self.fc5   = Linear(128, 10)

    def forward(self, x, training=True):
        x = self.fc1.forward(x)
        x = self.bn1.forward(x, training)
        x = self.relu1.forward(x)
        x = self.drop1.forward(x, training)

        x = self.fc2.forward(x)
        x = self.bn2.forward(x, training)
        x = self.relu2.forward(x)
        x = self.drop2.forward(x, training)

        x = self.fc3.forward(x)
        x = self.bn3.forward(x, training)
        x = self.relu3.forward(x)
        x = self.drop3.forward(x, training)

        x = self.fc4.forward(x)
        x = self.bn4.forward(x, training)
        x = self.relu4.forward(x)

        x = self.fc5.forward(x)
        return softmax(x)

    def backward(self, grad):
        grad = self.fc5.backward(grad)

        grad = self.relu4.backward(grad)
        grad = self.bn4.backward(grad)
        grad = self.fc4.backward(grad)

        grad = self.drop3.backward(grad)
        grad = self.relu3.backward(grad)
        grad = self.bn3.backward(grad)
        grad = self.fc3.backward(grad)

        grad = self.drop2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.bn2.backward(grad)
        grad = self.fc2.backward(grad)

        grad = self.drop1.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.bn1.backward(grad)
        grad = self.fc1.backward(grad)


# 参数更新
def update(model, lr):
    model.fc1.adam_update(lr)
    model.bn1.adam_update(lr)
    model.fc2.adam_update(lr)
    model.bn2.adam_update(lr)
    model.fc3.adam_update(lr)
    model.bn3.adam_update(lr)
    model.fc4.adam_update(lr)
    model.bn4.adam_update(lr)
    model.fc5.adam_update(lr)


# 余弦退火学习率
def cosine_lr(epoch, total_epochs, lr_max=1e-3, lr_min=1e-5):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / total_epochs))


# 训练
def train(model, X, y, epochs=30, lr=1e-3, batch_size=256):
    for epoch in range(epochs):
        lr_cur = cosine_lr(epoch, epochs, lr_max=lr, lr_min=1e-5)

        idx = np.random.permutation(len(X))
        X = X[idx]
        y = y[idx]

        total_loss = 0

        for i in range(
            0,
            len(X),
            batch_size
        ):
            Xb = X[i:i+batch_size]
            yb = y[i:i+batch_size]

            pred = model.forward(Xb, training=True)
            loss = cross_entropy(
                pred,
                yb
            )
            total_loss += loss
            grad = grad_loss(
                pred,
                yb
            )
            model.backward(grad)
            update(model, lr_cur)

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"lr={lr_cur:.6f}, "
            f"Loss={total_loss:.4f}"
        )


# 测试
def evaluate(model, X, y):
    pred = model.forward(X, training=False)

    pred_label = np.argmax(
        pred,
        axis=1
    )
    acc = np.mean(
        pred_label == y
    )
    print(
        "\nTest Accuracy:",
        acc
    )
    cm = confusion_matrix(
        y,
        pred_label
    )

    disp = ConfusionMatrixDisplay(cm)
    disp.plot()

    plt.show()


# 主程序
model = NeuralNetwork()
train(
    model,
    X_train,
    y_train
)
evaluate(
    model,
    X_test,
    y_test
)