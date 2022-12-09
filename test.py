# %%
# weight_decay = 0
# get_lbe = lambda: LBE(X.shape[1], kind="LF")

weight_decay = 1e-3
get_lbe = lambda: LBE(X.shape[1], kind="MLP")

# weight_decay = 0
# get_lbe = lambda: LBE_alternative(X.shape[1])

dataset = 'vote'
# dataset = 'madelon'
M_step_iters = 20

print_msg = False
kappa = 10
lr = 1e-3
pre_epochs = 100
epochs = 100

import torch
import os
from scipy.io import arff
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

dir_path = os.path.dirname(os.path.realpath(__file__))

def load_dataset(name):
    if name == "madelon":
        X = pd.concat([
            pd.read_csv("data/madelon_train.data", sep=" ", header=None).dropna(axis=1),
            pd.read_csv("data/madelon_valid.data", sep=" ", header=None).dropna(axis=1),
        ])
        y = pd.concat([
            pd.read_csv("data/madelon_train.labels", sep=" ", header=None).dropna(axis=1),
            pd.read_csv("data/madelon_valid.labels", sep=" ", header=None).dropna(axis=1),
        ])[0]
        y = np.where(y == 1, 1, 0)
    else:
        data = arff.loadarff(os.path.join(dir_path, 'data', f'{name}.arff'))
        df = pd.DataFrame(data[0])

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    return X, y

X, y = load_dataset(dataset)

X = X.fillna(X.median())
scaler = StandardScaler()
X = scaler.fit_transform(X)

X, y = torch.tensor(X), torch.tensor(y)

kfold = KFold(n_splits=5, shuffle=True)
dataset_split = list(kfold.split(X))

for pi in [0.2, 0.3, 0.4]:
    accs = []
    for fold, (train_ids, test_ids) in enumerate(dataset_split):
        X_train, y_train = X[train_ids], y[train_ids]
        X_test, y_test = X[test_ids], y[test_ids]

        n = len(y_train)

        from sklearn.linear_model import LogisticRegression
        import numpy as np

        clf = LogisticRegression()
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_train)
        np.mean(y_pred == y_train.numpy())


        def eta(x, lgr_param, intercept, kappa=10):
            return torch.pow(1 / (1 + torch.exp(-(x.double() @ lgr_param.T + intercept))), kappa)

        propensity = eta(
                X_train, 
                torch.tensor(clf.coef_, dtype=torch.double), 
                torch.tensor(clf.intercept_, dtype=torch.double),
                kappa = kappa,
            ).reshape(-1).double()
        propensity[torch.where(y_train == 0)] = 0
        propensity

        weights = propensity / propensity.sum()

        c = (1 - pi)
        n_pos = torch.sum(y_train).numpy()
        selected = np.random.choice(range(n), replace = False, size = int(c * n_pos), p = weights)
        # selected
        s = torch.zeros_like(y_train)
        s[selected] = 1
        s

        import torch.optim as optim
        
        from model import LBE
        from model import LBE_alternative
        lbe = get_lbe().cuda()

        X_train = X_train.float().cuda()
        y_train = y_train.cuda()
        X_test = X_test.cuda()
        y_test = y_test.cuda()
        s = s.float().cuda()

        lbe.pre_train(X_train, s, epochs=pre_epochs, lr=lr, print_msg=print_msg)

        s_pred = lbe.h(X_train)
        if print_msg:
            print("Accuracy:", ((s_pred.squeeze() > 0.5) == s).float().mean().item())

        optimizer = optim.Adam(lbe.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(epochs):
            P_y_hat = lbe.E_step(X_train, s)

            for M_step_iter in range(M_step_iters):
                optimizer.zero_grad()

                # Forward pass
                loss, _, _ = lbe.loss(X_train, s, P_y_hat)
                # Backward pass
                loss.backward()

                optimizer.step()

                if print_msg:
                    print('Epoch {}, step {}: train loss: {}'
                        .format(epoch, M_step_iter, loss.item()))
        
        with torch.no_grad():
            X_test = X_test.float()
            y_pred = lbe(X_test)
            accuracy = ((y_pred.squeeze() > 0.5) == y_test).float().mean().item()
            print(f"    FOLD {fold + 1} accuracy: {accuracy}")

            accs.append(accuracy)

    print(f"Pi: {pi}, accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")

# %%
