"""
Boucles d'apprentissages de critiques et acteurs. 
"""
import torch
from src.ppo.networks import CriticNet, ActorNet
from src.ppo.losses import Loss
import time

def fit_critic(model : CriticNet, optimizer, states : torch.Tensor, r_list : torch.Tensor,
                gamma : float, n_epochs = 100, verbose = False): 
    """
    Boucle d'entrainement du critique contre la perte (en MSE entre valeur prédite et return)

    Paramètres 
    ----------
    model : CriticNet 
    optimizer : 
    states : tensor (T, state_dim)
        Etats
    r_list : tensor (T, )
        Rewards
    gamma : float
        facteur d'actualisation
    n_epochs : int = 100
        nombre de passes sur les données
    verbose : bool = False
        indications visuelles durant l'entrainement
    
    Retourne
    --------
    l_history : list (n_epochs)
        historique de la perte le long de l'entrainement
    total_time : float
        temps total d'entrainement dans la boucle (pour les resultats numériques) 
    """

    l_history = []
    returns = Loss.compute_returns(r_list, gamma)
    model.train()

    t0 = time.time()
    for epoch in range (n_epochs):
        
        optimizer.zero_grad()

        loss = Loss.critic_loss_fn(model, states, returns)
        loss.backward()
        optimizer.step()

        if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
            print(
                f"[fit_critic] epoch={epoch+1}/{n_epochs} "
                f"loss={loss.item():.4e}"
            )
        l_history.append(loss.item())
    total_time = time.time()-t0
    return l_history, total_time

def fit_actor(model : ActorNet, optimizer, states : torch.Tensor, actions, avantages, n_epochs : int = 100, verbose : bool = False):
    """
    Boucle d'entrainement de l'acteur contre la perte (moyenne des log_proba pondérée par les avantages)

    Paramètres 
    ----------
    model : ActorNet 
    optimizer 
    states : tensor (T, state_dim)
        Etats
    actions : tensor(T,action_dim)
    
    avantages : tensor(T,)
    n_epochs : int = 100
        nombre de passes sur les données
    verbose : bool = False
        indications visuelles durant l'entrainement
    
    Retourne
    --------
    l_history : list (n_epochs)
        historique de la perte le long de l'entrainement
    total_time : float
        temps total d'entrainement dans la boucle (pour les resultats numériques) 
    """
    l_history = []
    t0 = time.time()
    model.train()

    for epoch in range(n_epochs): 

        optimizer.zero_grad()

        log_prob, _ = model.evaluate_actions(states, actions)
        loss = Loss.actor_loss_fn(log_prob, avantages)

        loss.backward()
        optimizer.step()
        if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
            print(
                f"[fit_actor] epoch={epoch+1}/{n_epochs} "
                f"loss={loss.item():.4e}"
            )
        l_history.append(loss.item())
    elapsed_time = time.time() - t0
    return l_history, elapsed_time

def fit_actor_ppo(model : ActorNet, optimizer, states : torch.Tensor, actions, avantages,old_log_prob, n_epochs : int = 100, verbose : bool = False):
    """
    Boucle d'entrainement de l'acteur contre la perte PPO (moyenne des ratios des log probas, pondérées par les avantages)

    Paramètres 
    ----------
    model : ActorNet 
    optimizer 
    states : tensor (T, state_dim)
        Etats
    actions : tensor(T,action_dim)
    
    avantages : tensor(T,)
        avantages collectés le long de la trajectoire
    old_log_prob : tensor(T,)
        log_probabilité collectée avec la trajectoire
    n_epochs : int = 100
        nombre de passes sur les données
    verbose : bool = False
        indications visuelles durant l'entrainement
    
    Retourne
    --------
    l_history : list (n_epochs)
        historique de la perte le long de l'entrainement
    total_time : float
        temps total d'entrainement dans la boucle (pour les resultats numériques) 
    """
    l_history = []
    t0 = time.time()
    model.train()
    for epoch in range(n_epochs): 

        optimizer.zero_grad()

        log_prob, _ = model.evaluate_actions(states, actions)
        loss = Loss.loss_actor_ppo_fn(log_prob,old_log_prob, avantages)

        loss.backward()
        optimizer.step()
        if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
            print(
                f"[fit_actor] epoch={epoch+1}/{n_epochs} "
                f"loss={loss.item():.4e}"
            )
        l_history.append(loss.item())
    elapsed_time = time.time() - t0
    return l_history, elapsed_time