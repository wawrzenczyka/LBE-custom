# %%
import torch

n = 2000

X1_neg = torch.normal(-2.7, 1, size=(int(n/2), 1))
X2_neg = torch.normal(0, 1, size=(int(n/2), 1))
X_neg = torch.cat([X1_neg, X2_neg], axis=1)

X1_pos = torch.normal(2.7, 1, size=(int(n/2), 1))
X2_pos = torch.normal(0, 1, size=(int(n/2), 1))
X_pos = torch.cat([X1_pos, X2_pos], axis=1)

X = torch.cat([X_neg, X_pos], axis=0)
y = torch.cat([torch.zeros(int(n/2)), torch.ones(int(n/2))], axis=0)

X, y

# %%
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")

# %%
from sklearn.linear_model import LogisticRegression
import numpy as np

clf = LogisticRegression()
clf.fit(X, y)

y_pred = clf.predict(X)
np.mean(y_pred == y.numpy())

# %%
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")

plt.show()

# %%
def eta(x, lgr_param, intercept, kappa=10):
    return torch.pow(1 / (1 + torch.exp(-(x.double() @ lgr_param.T + intercept))), kappa)

propensity = eta(
        X, 
        torch.tensor(clf.coef_, dtype=torch.double), 
        torch.tensor(clf.intercept_, dtype=torch.double),
        kappa = 1000,
    ).reshape(-1).double()
propensity[torch.where(y == 0)] = 0
propensity

# %%
weights = propensity / propensity.sum()

c = 0.4 * 0.5
selected = np.random.choice(range(n), replace = False, size = int(c * n), p = weights)
# selected
s = torch.zeros_like(y)
s[selected] = 1
s

# %%
import matplotlib.pyplot as plt
scatter = plt.scatter(X[:, 0], X[:, 1], c=s, s=20, edgecolor="k")
legend1 = plt.legend(*scatter.legend_elements(), title="Observed")

# %%
import torch.optim as optim
from model import LBE_MLP

lbe = LBE_MLP(2)
lbe.pre_train(X, s)

s_pred = lbe.h(X)
torch.sum((s_pred.reshape(-1) > 0.5) == s) / len(s)

# %%
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = (lbe.h(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)) > 0.5).int()
Z = Z.reshape(xx.shape).detach().numpy()

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=s, s=20, edgecolor="k")

plt.show()

# %%
optimizer = optim.Adam(lbe.parameters())

epoch = 1000
for epoch in range(epoch):
    optimizer.zero_grad()

    # Forward pass
    loss = lbe.loss(X, s)
    # Backward pass
    loss.backward()

    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
    optimizer.step()

# %%
y_pred = lbe.h(X)
torch.sum((y_pred.reshape(-1) > 0.5) == y) / len(y)

# %%
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = (lbe.h(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)) > 0.5).int()
Z = Z.reshape(xx.shape).detach().numpy()

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")

plt.show()

# %%
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = (lbe.eta(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)))
Z = Z.reshape(xx.shape).detach().numpy()

plt.contourf(xx, yy, Z, alpha=0.4)

plt.show()
