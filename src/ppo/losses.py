"""
Module de calcul des pertes utiles 
"""
import torch
from src.ppo.networks import ActorNet, CriticNet

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
        V = model(s).reshape(-1)
        G = G.reshape(-1)
        A = G - V
        return A.pow(2).mean()
    

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
        advantages = A.reshape(-1).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        loss = -(log_prob * advantages).mean()
        return loss
    
    @staticmethod
    def actor_loss_ppo_fn(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    A: torch.Tensor,
    entropy: torch.Tensor | None = None,
    eps_clip: float = 0.25,
    entropy_coef: float = 0.0
    ):
        """
        Perte PPO de l'acteur avec clipping du probability ratio.
        On définit :
            r_t(theta) = exp(log pi_theta - log pi_old)
        puis la surrogate objective clipée :
            L_t = min(r_t * A_t,clip(r_t, 1-eps, 1+eps) * A_t)
        La fonction retourne la quantité à minimiser :
        loss = - E[L_t]
        avec possibilité d'ajouter un bonus d'entropie.
        Paramètres
        ----------
        log_prob : tensor (T,) ou (T,1)
            Log-probabilité des actions sous la politique courante.
        old_log_prob : tensor (T,) ou (T,1)
            Log-probabilité des mêmes actions sous la politique de collecte.
        A : tensor (T,)
            Avantages estimés (GAE ou autre).
        entropy : tensor (T,) ou (T,1), optional
            Entropie de la politique courante.
        eps_clip : float
            Paramètre de clipping PPO.
        entropy_coef : float
            Poids du bonus d'entropie.

        Retourne
        --------
        loss : tensor scalaire
            Perte PPO à minimiser.
        """
        # mise en forme
        log_prob = log_prob.reshape(-1)
        old_log_prob = old_log_prob.reshape(-1)

        advantages = A.reshape(-1).detach()

        # normalisation classique PPO
        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std() + 1e-8)

        # ratio PPO
        log_ratio = log_prob - old_log_prob
        r_t = torch.exp(log_ratio)

        # clipping
        clipped = torch.clamp(r_t, 1.0 - eps_clip, 1.0 + eps_clip)

        # surrogate objective
        L = torch.min(
            r_t * advantages,
            clipped * advantages
        )

        # loss acteur
        loss = -L.mean()

        # bonus d'entropie optionnel
        if entropy is not None:
            entropy = entropy.reshape(-1)
            loss = loss - entropy_coef * entropy.mean()

        return loss
    
    @staticmethod
    def compute_td_residual(rewards : torch.Tensor, values : torch.Tensor, next_values: torch.Tensor, 
                             dones: torch.Tensor, gamma : float): 
        """
        Calcule les résidus temporels TD :
        delta_t = r_t + gamma (1-d_t) V(s_{t+1}) - V(s_t)
        Paramètres
        ----------
        rewards : tensor (T,)
        values : tensor (T,)
        next_values : tensor (T,)
        dones : tensor (T,)
            1 si terminal, 0 sinon
        gamma : float

        Retourne
        --------
        delta : tensor (T,)
        """
        values = values.reshape(-1)
        delta = rewards + gamma * (1-dones)*next_values - values
        return delta
    
    @staticmethod
    def compute_gae(rewards : torch.Tensor, values : torch.Tensor, next_values : torch.Tensor,
                    dones : torch.Tensor, gamma : float, lam : float):
        """
        Calcule les avantages GAE
        """
        delta = Loss.compute_td_residual(rewards, values, next_values, dones, gamma)
        gae = torch.zeros_like(delta)
        gae[-1] = delta[-1]
        for i in reversed(range(len(gae)-1)): 
            gae[i] = delta[i] + gamma*lam*(1-dones[i])*gae[i+1]
        ### Normalisation
        gae = (gae - gae.mean())/(gae.std()+1e-12)
        return gae
    
    @staticmethod
    def compute_ppo_returns(advantages : torch.Tensor, values : torch.Tensor):
        """
        Construit les cibles du critic a partir des avantages PPO
        """
        advantages = advantages.reshape(-1)
        values = values.reshape(-1)
        return advantages + values
    




    

