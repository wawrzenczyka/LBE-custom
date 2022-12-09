# %%
import torch

n = 2000
n_pos = int(n/2)

# X1_neg = torch.normal(-2.7, 1, size=(n_pos, 1))
# X1_pos = torch.normal(2.7, 1, size=(n_pos, 1))
# X1 = torch.cat([X1_neg, X1_pos], axis=0)
# X2 = torch.normal(0, 1, size=(n, 1))

X1 = torch.FloatTensor(n, 1).uniform_(-5, 5)
X2 = torch.FloatTensor(n, 1).uniform_(-5, 5)

X = torch.cat([X1, X2], axis=1)
X_bias = torch.cat([X, torch.ones_like(X1)], axis=1)

beta = torch.tensor([10, 0, 0], dtype=torch.float)
gamma = torch.tensor([0, 10, 0], dtype=torch.float)

y = torch.bernoulli(torch.sigmoid(X_bias @ beta))
s = torch.bernoulli(torch.sigmoid(X_bias @ gamma))
s = torch.where((s == 1) & (y == 1), 1, 0)
print(y)

# %%
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")

# %%
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=s, s=20, edgecolor="k")

# %%
import numpy as np
import torch.optim as optim

from model import LBE
from model import LBE_alternative
# lbe = LBE(2, kind="LF")
lbe = LBE(2, kind="MLP")
# lbe = LBE_alternative(2)

X = X.float()
s = s.float()
lbe.pre_train(X, s, epochs=500, lr=1e-2)

s_pred = lbe(X)
print(torch.sum((s_pred.squeeze() > 0.5) == s) / len(s))
# print(lbe.theta_h)

# # %%
# from sklearn.linear_model import LogisticRegression
# import numpy as np

# X = X.float()
# s = s.float()

# clf = LogisticRegression()
# clf.fit(X, s)

# s_pred = clf.predict(X)
# print(np.mean(s_pred == s.numpy()))
# print(clf.coef_, clf.intercept_)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = (lbe.h(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)) > 0.5).int()
Z = Z.reshape(xx.shape).detach().numpy()

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=s, s=20, edgecolor="k")

plt.show()


# %%
from torch.optim.lr_scheduler import StepLR

optimizer = optim.Adam(lbe.parameters(), lr=1e-3)
# scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

epochs = 100
for epoch in range(epochs):
    P_y_hat = lbe.E_step(X, s)

    for M_step_iter in range(100):
        optimizer.zero_grad()

        # Forward pass
        loss, grad_theta_1, grad_theta_2 = lbe.loss(X, s, P_y_hat)
        # Backward pass
        loss.backward()

        optimizer.step()

        # print(grad_theta_1, lbe.theta_h.grad.squeeze())
        # print(grad_theta_2, lbe.theta_eta.grad.squeeze())

        # print('Epoch {}, step {}: train loss: {}'
        #     .format(epoch, M_step_iter, loss.item()))
    
    print('Epoch {}: train loss: {}'
        .format(epoch, loss.item()))
    print(f"theta_h: {lbe.get_theta_h()}, true: {beta}")
    print(f"theta_eta: {lbe.get_theta_eta()}, true: {gamma}")
    # scheduler.step()

y_pred = lbe(X)
torch.sum((y_pred.squeeze() > 0.5) == y) / len(y)

# %%
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = (lbe(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)) > 0.5).int()
Z = Z.reshape(xx.shape).detach().numpy()

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")

plt.show()


# %%
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = lbe.eta(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32), sigmoid=True)
Z = Z.reshape(xx.shape).detach().numpy()

plt.figure(figsize=(8, 6))
contour = plt.contourf(xx, yy, Z, alpha=0.4, cmap="Blues", levels=np.linspace(0, 1, 11))
plt.colorbar(contour)

plt.show()

# %%
print(f"theta_h: {lbe.get_theta_h()}, true: {beta}")
print(f"theta_eta: {lbe.get_theta_eta()}, true: {gamma}")

# %%
