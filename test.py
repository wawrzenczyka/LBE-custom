# %%
dataset = 'vote'
kappa = 10
# pi = 0.4
pre_epochs = 100
epochs = 100
M_step_iters = 10
lr = 1e-3
weight_decay = 1e-2
print_msg = False

for pi in [0.2, 0.3, 0.4]:
    import torch
    import os
    import re
    from scipy.io import arff
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold


    dir_path = os.path.dirname(os.path.realpath(__file__))


    def read_names_file(filename):
        with open(filename, 'r') as f:
            columns = []
            while True:
                s = f.readline()
                if s == '':
                    break

                match = re.match(r'([^:]+):\s+[a-zA-Z]+\.', s)
                
                if match is not None:
                    column_name = match.groups()[0]
                    columns.append(column_name)
                
            return columns


    def get_datasets():
        names = [
            'Adult',
            'BreastCancer',
            'credit-a',
            'credit-g',
            'diabetes',
            'heart-c',
            'spambase',
            'vote',
            'wdbc',
        ]

        return {name: load_dataset(name) for name in names}


    def load_dataset(name):
        data = arff.loadarff(os.path.join(dir_path, 'data', f'{name}.arff'))
        df = pd.DataFrame(data[0])

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        return X.to_numpy(), y.to_numpy()

    X, y = load_dataset(dataset)
    #Obtain mean of columns as you need, nanmean is convenient.
    col_mean = np.nanmean(X, axis=0)
    #Find indices that you need to replace
    inds = np.where(np.isnan(X))
    #Place column means in the indices. Align the arrays using take
    X[inds] = np.take(col_mean, inds[1])

    X, y = torch.tensor(X), torch.tensor(y)

    kfold = KFold(n_splits=5, shuffle=True)

    accs = []
    for fold, (train_ids, test_ids) in enumerate(kfold.split(X)):
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
        # lbe = LBE(X.shape[1], kind="LF")
        lbe = LBE(X.shape[1], kind="MLP")
        # lbe = LBE_alternative(X.shape[1])

        X_train = X_train.float()
        s = s.float()
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
            y_pred = lbe.h(X_test)
            accuracy = ((y_pred.squeeze() > 0.5) == y_test).float().mean().item()
            print(f"    FOLD {fold + 1} accuracy: {accuracy}")

            accs.append(accuracy)

    print(f"Pi: {pi}, mean accuracy: {np.mean(accs)}")