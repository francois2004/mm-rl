"""
Contient les définitions des réseaux de neurones utilisés pour l'actor critic
learning
"""

import torch
from torch import nn 
from torch import distributions


class ActorNet(nn.Module) :
    """
    Politique continue scalaire pour choisir un demi-spread delta dans
    [delta_min, delta_max].

    Backbone fully connected avec activations tanh.
    La politique est définie sur une variable latente gaussienne,
    ensuite transformée par tanh puis remise à l'échelle.
    """
    def __init__(
        self,
        state_dim: int = 4,
        hidden_size: int = 64,
        n_layers: int = 3,
        action_dim: int = 1,
        delta_min: float = 0.0,
        delta_max: float = 0.05,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        q_min = .0, 
        q_max = 10.
    ):
        super().__init__()

        if n_layers < 1:
            raise ValueError("n_layers doit être >= 1")
        if action_dim not in [1, 4]:
            raise ValueError("action_dim doit valoir 1 ou 4")
        if delta_max <= delta_min:
            raise ValueError("Il faut delta_max > delta_min.")

        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.action_dim = action_dim
        self.delta_min = float(delta_min)
        self.delta_max = float(delta_max)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.q_min = q_min
        self.q_max = q_max

        layers = [nn.Linear(state_dim, hidden_size), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)

        self.mu_head = nn.Linear(hidden_size, action_dim)
        self.log_std_head = nn.Linear(hidden_size, action_dim)

    
    def forward(self, s : torch.Tensor): 
        """
        prend en entrée l'état s du marché,
        renvoie les paramètres latents de la politique: 
            mu : moyenne de la gaussienne latente
            log_std : log ecart_type, tronqué pour stabilité
        """
        h = self.backbone(s)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    
    def _latent_dist(self, s : torch.Tensor): 
        """
        distribution gaussienne latente 
        """
        mu, log_std = self.forward(s)
        std = log_std.exp()
        return distributions.Normal(mu, std)
    
    def evaluate_actions(self, s: torch.Tensor, action: torch.Tensor):
        """
        Évalue des actions déjà prises, utile pour PPO.
        Paramètres
        ----------
        s : tensor (batch, state_dim)
            Etats
        action : tensor (batch, action_dim)
            Actions bornées

        Retourne
        --------
        log_prob : tensor (batch, 1)
            Log-probabilité de l'action
        entropy : tensor (batch, 1)
            Entropie de la distribution latente
        """
        eps = 1e-6

        if self.action_dim == 1:
            a01 = (action - self.delta_min) / (self.delta_max - self.delta_min)
            a01 = torch.clamp(a01, eps, 1.0 - eps)

        elif self.action_dim == 4:
            a01 = torch.zeros_like(action)

            a01[..., 0:1] = (action[..., 0:1] - self.delta_min) / (self.delta_max - self.delta_min)
            a01[..., 1:2] = (action[..., 1:2] - self.delta_min) / (self.delta_max - self.delta_min)
            a01[..., 2:3] = (action[..., 2:3] - self.q_min) / (self.q_max - self.q_min)
            a01[..., 3:4] = (action[..., 3:4] - self.q_min) / (self.q_max - self.q_min)

            a01 = torch.clamp(a01, eps, 1.0 - eps)

        else:
            raise ValueError(f"action_dim inattendu : {self.action_dim}")

        u = 2.0 * a01 - 1.0
        u = torch.clamp(u, -1.0 + eps, 1.0 - eps)

        z = 0.5 * torch.log((1.0 + u) / (1.0 - u))

        dist = self._latent_dist(s)
        log_prob_z = dist.log_prob(z).sum(dim=-1, keepdim=True)
        log_det_jac = torch.log(1.0 - u.pow(2) + eps).sum(dim=-1, keepdim=True)
        log_prob = log_prob_z - log_det_jac

        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy
    
    def _squash_action(self, z):

        """
    Transforme un latent gaussien en action bornée.
    Paramètres
    ----------
    z : torch.Tensor
        Variable latente issue de la distribution.
    Retourne
    --------
    action : torch.Tensor
        Action bornée.
    """
        u = torch.tanh(z)
        a01 = (u + 1.0) / 2.0
        if self.action_dim == 1:
            delta = self.delta_min + (self.delta_max - self.delta_min) * a01
            return delta
        elif self.action_dim == 4:
            delta_bid = self.delta_min + (self.delta_max - self.delta_min) * a01[..., 0:1]
            delta_ask = self.delta_min + (self.delta_max - self.delta_min) * a01[..., 1:2]
            q_bid = self.q_min + (self.q_max - self.q_min) * a01[..., 2:3]
            q_ask = self.q_min + (self.q_max - self.q_min) * a01[..., 3:4]  
            action = torch.cat([delta_bid, delta_ask, q_bid, q_ask], dim=-1)
            return action
        else:
            raise ValueError(f"action_dim inattendu : {self.action_dim}")

    def sample_action(self, state : torch.Tensor): 
        """
        Construit une action suivant la distribution de la politique, en fonction de l'etat du marché. 

        Paramètres
        ----------
        state : torch.Tensor (state_dim) ou (batch, state_dim)
            etat du marché

        Retourne 
        --------
        delta : torch.tensor (action_dim)
            Action échantillonnée (après transformation)
        log_prob : shape [batch, 1]
            Log_probabilité de l'action
        entropy  : shape [batch, 1]
            Entropie de la distribution
        """
        distrib = self._latent_dist(state)
        z = distrib.rsample()
        u = torch.tanh(z)
        action = self._squash_action(z)
        eps = 1e-6

        log_prob_z = distrib.log_prob(z).sum(dim = -1, keepdim=True)
        log_det_jac = torch.log(1.0 - u.pow(2) + eps).sum(dim=-1, keepdim=True)
        log_prob = log_prob_z - log_det_jac

        entropy = distrib.entropy().sum(dim=-1, keepdim=True)

        return action, log_prob, entropy

    

class CriticNet(nn.Module):
    def __init__(self, state_dim=4, hidden_size=64, n_layers=3):
        super().__init__()
        layers = [nn.Linear(state_dim, hidden_size), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers += [nn.Linear(hidden_size, 1)]
        self.network = nn.Sequential(*layers)

    def forward(self, s):
        return self.network(s)

