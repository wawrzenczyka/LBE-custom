import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=10):
        super(Classifier, self).__init__()
        self.h = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.h(x)


class PropensityEstimator(nn.Module):
    def __init__(self, input_dim, hidden_dim=10):
        super(PropensityEstimator, self).__init__()
        self.eta = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # for layer in [module for module in self.eta.modules()
        #                 if isinstance(module, nn.Linear)]:
        #     layer.weight.data.fill_(0)
        #     layer.bias.data.fill_(0)
    
    def forward(self, x):
        return self.eta(x)


class LBE_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=10):
        super(LBE_MLP, self).__init__()
        self.h = Classifier(input_dim, hidden_dim)
        self.eta = PropensityEstimator(input_dim, hidden_dim)
    
    def forward(self, x):
        h = self.h(x)
        return h

    def loss(self, x, s):
        h = torch.clamp(self.h(x), 1e-5, 1 - 1e-5).squeeze()
        eta = torch.clamp(self.eta(x), 1e-5, 1 - 1e-5).squeeze()

        s = s

        # L_i_1 = h * eta
        # L_i_0 = (1 - h) * eta

        # P_s_given_x = L_i_1 + L_i_0
        
        P_y_hat_1 = torch.where(s == 1, eta, 1 - eta) * h
        P_y_hat_0 = torch.where(s == 1, 0, 1) * (1 - h)

        P_y_hat = torch.cat([P_y_hat_0.reshape(-1, 1), P_y_hat_1.reshape(-1, 1)], axis = 1)
        P_y_hat /= P_y_hat.sum(axis=1).reshape(-1, 1)

        loss = torch.where(
            s == 1,
            P_y_hat[:, 1] * (torch.log(h) + torch.log(eta)),
            P_y_hat[:, 1] * (torch.log(h) + torch.log(1 - eta)) + P_y_hat[:, 0] * torch.log(1 - h),
        )

        return -torch.mean(loss)

    def pre_train(self, x, s):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.h.parameters())

        epoch = 100
        for epoch in range(epoch):
            optimizer.zero_grad()
            # Forward pass
            y_pred = self.h(x)
            # Compute Loss
            loss = criterion(y_pred.squeeze(), s)
        
            print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
            # Backward pass
            loss.backward()
            optimizer.step()
