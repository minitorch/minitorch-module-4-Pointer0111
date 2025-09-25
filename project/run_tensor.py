"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch


def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()

        # Submodules
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        # 第一层：输入 -> 隐藏层，然后应用 ReLU
        middle = self.layer1.forward(x).relu()
        # 第二层：隐藏层 -> 隐藏层，然后应用 ReLU
        end = self.layer2.forward(middle).relu()
        # 第三层：隐藏层 -> 输出层，然后应用 Sigmoid
        return self.layer3.forward(end).sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        # 调试：打印形状
        # print('x.shape=', x.shape, 'W.shape=', self.weights.value.shape, 'b.shape=', self.bias.value.shape)
        # 使用广播乘法与求和代替矩阵乘法：x @ weights + bias
        W = self.weights.value
        b = self.bias.value
        xc = x.contiguous()
        Wc = W.contiguous()
        # print('sizes: x.size=', xc.size, 'W.size=', Wc.size)
        n, m = xc.shape
        mi, mo = Wc.shape
        # print('n,m,mi,mo =', n, m, mi, mo, 'prod x=', n*m, 'target=', n*m*1)
        x3 = xc.view(n, m, 1)
        W3 = Wc.view(1, mi, mo)
        y = (x3 * W3).sum(dim=1).view(n, mo)
        return y + b.view(1, b.shape[0])


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        print('start training: N=', data.N, 'X.shape=', X.shape, 'y.shape=', y.shape)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            # print('epoch', epoch)
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
