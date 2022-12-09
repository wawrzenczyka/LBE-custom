import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Classifier(nn.Module):
    def forward(self, x):
        pass


class MLPClassifier(Classifier):
    def __init__(self, input_dim, hidden_dim=10):
        super(MLPClassifier, self).__init__()
        self.h = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            # nn.Sigmoid(),
        )
    
    def forward(self, x, sigmoid=False):
        h = self.h(x)
        if sigmoid:
            h = torch.sigmoid(h)
        return h


class LogisticClassifier(Classifier):
    def __init__(self, input_dim):
        super(LogisticClassifier, self).__init__()
        self.h = nn.Sequential(
            nn.Linear(input_dim, 1),
            # nn.Sigmoid(),
        )
    
    def forward(self, x, sigmoid=False):
        h = self.h(x)
        if sigmoid:
            h = torch.sigmoid(h)
        return h


class PropensityEstimator(nn.Module):
    def forward(self, x):
        pass


class MLPPropensityEstimator(PropensityEstimator):
    def __init__(self, input_dim, hidden_dim=10):
        super(MLPPropensityEstimator, self).__init__()
        self.eta = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            # nn.Sigmoid(),
        )

        for layer in [module for module in self.eta.modules()
                        if isinstance(module, nn.Linear)]:
            # layer.weight = nn.Parameter(torch.randn_like(layer.weight) / 100)
            # layer.bias = nn.Parameter(torch.randn_like(layer.bias) / 100)
            layer.weight = nn.Parameter(torch.zeros_like(layer.weight))
            layer.bias = nn.Parameter(torch.zeros_like(layer.bias))
    
    def forward(self, x, sigmoid=False):
        eta = self.eta(x)
        if sigmoid:
            eta = torch.sigmoid(eta)
        return eta


class LogisticPropensityEstimator(PropensityEstimator):
    def __init__(self, input_dim):
        super(LogisticPropensityEstimator, self).__init__()
        self.eta = nn.Sequential(
            nn.Linear(input_dim, 1),
            # nn.Sigmoid(),
        )

        for layer in [module for module in self.eta.modules()
                        if isinstance(module, nn.Linear)]:
            # layer.weight = nn.Parameter(torch.randn_like(layer.weight) / 100)
            # layer.bias = nn.Parameter(torch.randn_like(layer.bias) / 100)
            layer.weight = nn.Parameter(torch.zeros_like(layer.weight))
            layer.bias = nn.Parameter(torch.zeros_like(layer.bias))
    
    def forward(self, x, sigmoid=False):
        eta = self.eta(x)
        if sigmoid:
            eta = torch.sigmoid(eta)
        return eta


class LBE(nn.Module):
    def __init__(self, input_dim, kind="MLP", hidden_dim=10):
        super(LBE, self).__init__()
        if kind == "MLP":
            self.h = MLPClassifier(input_dim, hidden_dim)
            self.eta = MLPPropensityEstimator(input_dim, hidden_dim)
        elif kind == "LF":
            self.h = LogisticClassifier(input_dim)
            self.eta = LogisticPropensityEstimator(input_dim)
    
    def get_theta_h(self):
        value = torch.tensor([], dtype=torch.float)
        for param in self.h.parameters():
            value = torch.cat([value.squeeze(), param.data.reshape(-1)])
        return value
    
    def get_theta_eta(self):
        value = torch.tensor([], dtype=torch.float)
        for param in self.eta.parameters():
            value = torch.cat([value.squeeze(), param.data.reshape(-1)])
        return value

    def forward(self, x):
        h = self.h(x, sigmoid=True)
        return h

    def E_step(self, x, s):
        with torch.no_grad():
            h = self.h(x, sigmoid=True).squeeze()
            eta = self.eta(x, sigmoid=True).squeeze()

            P_y_hat_1 = torch.where(s == 1, eta, 1 - eta) * h
            P_y_hat_0 = torch.where(s == 1, 0, 1) * (1 - h)

            P_y_hat = torch.cat([P_y_hat_0.reshape(-1, 1), P_y_hat_1.reshape(-1, 1)], axis = 1)
            P_y_hat /= P_y_hat.sum(axis=1).reshape(-1, 1)
            return P_y_hat

    def loss(self, x, s, P_y_hat):
        h = self.h(x).squeeze()
        eta = self.eta(x).squeeze()

        log_h = F.logsigmoid(h)
        log_1_minus_h = F.logsigmoid(-h)
        log_eta = F.logsigmoid(eta)
        log_1_minus_eta = F.logsigmoid(-eta)

        # loss = torch.where(
        #     s == 1,
        #     P_y_hat[:, 1] * (log_h + log_eta) + 0,
        #     P_y_hat[:, 1] * (log_h + log_1_minus_eta) + P_y_hat[:, 0] * log_1_minus_h
        # )
        loss = torch.where(
            s == 1,
            P_y_hat[:, 1] * (log_h + log_eta) + P_y_hat[:, 0] * (log_1_minus_h + log_eta),
            P_y_hat[:, 1] * (log_h + log_1_minus_eta) + P_y_hat[:, 0] * (log_1_minus_h + log_1_minus_eta)
        )
        
        with torch.no_grad():
            x = torch.cat([x, torch.ones((len(x), 1))], dim=1)
            sigma_h = torch.sigmoid(h)
            sigma_eta = torch.sigmoid(eta)
            grad_theta_1 = ((P_y_hat[:, 0] * sigma_h).reshape(-1, 1) * x + (P_y_hat[:, 1] * (sigma_h - 1)).reshape(-1, 1) * x).sum(axis = 0)
            grad_theta_2 = (((-1)**(s+1) * (1 * P_y_hat[:, 1] / torch.where(s == 1, sigma_eta, 1 - sigma_eta)) * sigma_eta * (sigma_eta - 1)).reshape(-1, 1) * x).sum(axis = 0)

        return -torch.sum(loss), grad_theta_1, grad_theta_2

    def pre_train(self, x, s, epochs=100, lr=1e-3, print_msg=False):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.h.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass
            s_logits = self.h(x)
            # Compute Loss
            loss = criterion(s_logits.squeeze(), s)
            # Backward pass
            loss.backward()
            optimizer.step()

            if print_msg:
                print('Epoch {}: train loss: {}'.format(epoch, loss.item()))


class LBE_alternative(nn.Module):
    def __init__(self, input_dim):
        super(LBE_alternative, self).__init__()
        self.theta_h = nn.Parameter(torch.randn((input_dim + 1, 1)) / 100, requires_grad=True)
        self.theta_eta = nn.Parameter(torch.randn((input_dim + 1, 1)) / 100, requires_grad=True)

    def h(self, x, sigmoid=False):
        x = torch.cat([x, torch.ones((len(x), 1))], dim=1)
        h = x @ self.theta_h
        if sigmoid:
            h = torch.sigmoid(h)
        return h

    def eta(self, x, sigmoid=False):
        x = torch.cat([x, torch.ones((len(x), 1))], dim=1)
        eta = x @ self.theta_eta
        if sigmoid:
            eta = torch.sigmoid(eta)
        return eta
    
    def get_theta_h(self):
        return self.theta_h.data.squeeze()
    
    def get_theta_eta(self):
        return self.theta_eta.data.squeeze()

    def forward(self, x):
        h = self.h(x, sigmoid=True)
        return h

    def E_step(self, x, s):
        with torch.no_grad():
            h = self.h(x, sigmoid=True).squeeze()
            eta = self.eta(x, sigmoid=True).squeeze()

            P_y_hat_1 = torch.where(s == 1, eta, 1 - eta) * h
            P_y_hat_0 = torch.where(s == 1, 0, 1) * (1 - h)

            P_y_hat = torch.cat([P_y_hat_0.reshape(-1, 1), P_y_hat_1.reshape(-1, 1)], axis = 1)
            P_y_hat /= P_y_hat.sum(axis=1).reshape(-1, 1)
            return P_y_hat

    def loss(self, x, s, P_y_hat):
        h = self.h(x).squeeze()
        eta = self.eta(x).squeeze()

        log_h = F.logsigmoid(h)
        log_1_minus_h = F.logsigmoid(-h)
        log_eta = F.logsigmoid(eta)
        log_1_minus_eta = F.logsigmoid(-eta)

        loss = torch.where(
            s == 1,
            P_y_hat[:, 1] * (log_h + log_eta) + 0,
            P_y_hat[:, 1] * (log_h + log_1_minus_eta) + P_y_hat[:, 0] * log_1_minus_h
        )
        
        with torch.no_grad():
            x = torch.cat([x, torch.ones((len(x), 1))], dim=1)
            sigma_h = torch.sigmoid(h)
            sigma_eta = torch.sigmoid(eta)
            grad_theta_1 = ((P_y_hat[:, 0] * sigma_h).reshape(-1, 1) * x + (P_y_hat[:, 1] * (sigma_h - 1)).reshape(-1, 1) * x).sum(axis = 0)
            grad_theta_2 = (((-1)**(s+1) * (1 * P_y_hat[:, 1] / torch.where(s == 1, sigma_eta, 1 - sigma_eta)) * sigma_eta * (sigma_eta - 1)).reshape(-1, 1) * x).sum(axis = 0)

        return -torch.sum(loss), grad_theta_1, grad_theta_2

    def pre_train(self, x, s, epochs=100, lr=1e-3, print_msg=False):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam([self.theta_h], lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass
            s_logits = self.h(x)
            # Compute Loss
            loss = criterion(s_logits.squeeze(), s)
            # Backward pass
            loss.backward()
            optimizer.step()

            if print_msg:
                print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
