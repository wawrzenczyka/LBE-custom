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


# # %%
# from model import LogisticClassifier

# nn_clf = LogisticClassifier(2)
# criterion = torch.nn.BCELoss()

# optimizer = torch.optim.Adam(nn_clf.parameters())

# for epoch in range(1000):
#     optimizer.zero_grad() # Setting our stored gradients equal to zero
#     outputs = nn_clf(X)
#     loss = criterion(torch.squeeze(outputs), y) # [200,1] -squeeze-> [200]

#     loss.backward() # Computes the gradient of the given tensor w.r.t. graph leaves 
#     optimizer.step() # Updates weights and biases with the optimizer (SGD)

# y_proba = nn_clf(X)
# y_pred = torch.where(y_proba >.5, 1, 0).squeeze()
# print(torch.sum(y_pred == y) / len(y))

# for param in nn_clf.named_parameters():
#     print(param)
    

# # %%
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Z = nn_clf(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)).squeeze() > 0.5
# Z = Z.reshape(xx.shape)

# plt.contourf(xx, yy, Z, alpha=0.4)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")

# plt.show()

# %%
def eta(x, lgr_param, intercept, kappa=10):
    return torch.pow(1 / (1 + torch.exp(-(x.double() @ lgr_param.T + intercept))), kappa)

propensity = eta(
        X, 
        torch.tensor(clf.coef_, dtype=torch.double), 
        torch.tensor(clf.intercept_, dtype=torch.double),
        kappa = 100000,
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
from model import LBE

lbe = LBE(2, kind="LF")
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
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

epoch = 1000
for epoch in range(epoch):
    optimizer.zero_grad()

    # Forward pass
    loss = lbe.loss(X, s)
    # Backward pass
    loss.backward()

    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
    optimizer.step()
    
    scheduler.step()


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

# %%
