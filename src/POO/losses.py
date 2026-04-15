"""
Module de calcul des pertes utiles 
"""
import torch
from src.POO.networks import ActorNet, CriticNet

class Loss :
    """
    Fonctions utiles pour calculer les returns et la perte du critic.
    """ 
    @staticmethod
    def compute_returns(r_list : torch.Tensor, gamma: float):
        """
        Calcule les returns G_t à partir des rewards.

        G_t = r_t + gamma * G_{t+1}

        Paramètres
        ----------
        r_list : tensor (T,)
            Rewards de la trajectoire
        gamma : float
            Facteur d'actualisation

        Retourne
        -------
        tensor (T,)
           Returns
        """
        G = torch.zeros_like(r_list)
        G[-1] = r_list[-1]
        for i in range (1,len(G)): 
            G[-i-1] = gamma*G[-i] + r_list[-i-1]

        return G
    
    @staticmethod
    def critic_loss_fn(model : CriticNet, s: torch.Tensor,G: torch.Tensor): 
        """
        Perte du critic (MSE entre valeur prédite et return).

        L = mean((V(s_t) - G_t)^2)

        Parameters
        ----------
        model : CriticNet

        s : tensor (T, state_dim)
            États
        G : tensor(T,)
            Returns

        Retourne
        -------
        Loss
            Perte scalaire de l'acteur
        """
        V = model(s)
        residual = (V-G).reshape(-1)
        return residual.pow(2).mean()
    

    def compute_advantages(model : CriticNet, s : torch.Tensor, G: torch.Tensor): 
        """
        Calcule l'avantage de la politique actuelle relativement à la valeur estimée de l'état

        Paramètres 
        ----------
        model : CriticNet
            fonction de valeur apprise sur la politique récente
        s : tensor (batch, state_dim)
            Etat
        G : tensor (T,)
            Returns

        Retourne
        --------
        A : tensor(T,)
            Avantage de la trajectoire
        """
        V = model(s).reshape(-1)
        G = G.reshape(-1)
        A = G - V
        return A
    
    def actor_loss_fn(log_prob : torch.Tensor, A : torch.Tensor): 
        """
        Perte de l'acteur 

        Paramètres
        ----------
        log_prob : tensor(T,)
            log_prob des actions sous la politique courante 
        A : tensor (T,)
            Avantage le long de la trajectoire

        Retourne
        --------
        L : float
            Perte de l'acteur 
        """
        log_prob = log_prob.reshape(-1)
        avantages = A.reshape(-1).detach()

        loss = -(log_prob - avantages).mean()
        return loss

    

