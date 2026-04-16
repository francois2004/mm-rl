"""
algorithmes d'entrainement, mise en oeuvre
"""

import torch 

from src.ppo.networks import ActorNet, CriticNet
from src.envs.env_toy_mm import MMSimulator
from src.utils.device import get_device
from src.ppo.losses import Loss
import src.ppo.trainers as tr


device = get_device()

def collect_trajectory(env : MMSimulator, model : ActorNet, device = device, reset_rdm : bool = True, max_steps : int = 200):
    """
    Collecte un épisode complet sous la politique courante.

    Paramètre
    ---------
    env : MMSimulator
        simulateur de LOB
    model : ActorNet
    device : torch.device 
        accélerateur gpu (ou non)

    Retourne 
    --------
    states, actions, rewards, old log_prob
    """
    model.eval()

    # Initialisation de l'épisode
    if reset_rdm: 
        state = env.reset_random(max_steps)
    else : 
        state = env.reset()
    state = env._state()
    done = False

    states = []
    actions = []
    rewards = []
    old_log_probs = []

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action_tensor, log_prob, _ = model.sample_action(state_tensor)          # entropie encore non prise en compte 


        states.append(state_tensor.squeeze(0))          
        actions.append(action_tensor.squeeze(0))        
        old_log_probs.append(log_prob.squeeze(0))       

        action_env = action_tensor.squeeze(0).squeeze(-1).item()

        next_state, reward, done = env.step(action_env)

        rewards.append(reward)

        # 6) mise à jour de l'état
        state = next_state

    # Conversion finale en tenseurs
    trajectory = {
        "states": torch.stack(states, dim=0),                           # (T, state_dim)
        "actions": torch.stack(actions, dim=0),                         # (T, action_dim)
        "rewards": torch.tensor(rewards, dtype=torch.float32, device=device),  # (T,)
        "old_log_probs": torch.stack(old_log_probs, dim=0),             # (T, 1)
    }

    return trajectory

def train_one_episode(env : MMSimulator, actor : ActorNet, critic : CriticNet, 
                      actor_optimizer, critic_optimizer, device, gamma, 
                      n_epochs_actor = 10, n_epochs_critic = 10, random_reset = True, 
                      max_steps = 200, verbose = False):
    """
Exécute un épisode complet d'apprentissage Actor–Critic.

La fonction :
- collecte une trajectoire sous la politique courante ;
- calcule les returns à partir des rewards ;
- entraîne le critic pour approximer la fonction de valeur ;
- calcule les avantages ;
- entraîne l’acteur à partir de ces avantages.

Parameters
----------
env : MMSimulator
    Environnement de trading simulé.
actor : ActorNet
    Réseau de politique.
critic : CriticNet
    Réseau de valeur.
actor_optimizer : torch.optim.Optimizer
    Optimiseur de l’acteur.
critic_optimizer : torch.optim.Optimizer
    Optimiseur du critic.
device : torch.device
    Device utilisé pour les calculs.
gamma : float
    Facteur d’actualisation.
n_epochs_actor : int, optional
    Nombre d’itérations d’entraînement de l’acteur.
n_epochs_critic : int, optional
    Nombre d’itérations d’entraînement du critic.
random_reset : bool, optional
    Si True, démarre l’épisode à un point aléatoire.
max_steps : int, optional
    Longueur maximale de l’épisode.
verbose : bool, optional
    Affiche des informations pendant l’entraînement.

Returns
-------
l_history_critic : list
    Historique des pertes du critic sur l’épisode.
l_history_actor : list
    Historique des pertes de l’acteur sur l’épisode.
episode_return : float
    Somme des rewards sur l’épisode.
"""
    trajectory = collect_trajectory(env, actor, device, random_reset, max_steps)
    # Récupération
    states = trajectory["states"]
    actions = trajectory["actions"]
    rewards = trajectory["rewards"]
    #pour PPO
    old_log_probs = trajectory["old_log_probs"]

    G = Loss.compute_returns(rewards, gamma)
    if verbose :
        print('[train_critic]')
    l_history_critic, _ = tr.fit_critic(critic, critic_optimizer, states, rewards, gamma, n_epochs_critic, verbose)

    A = Loss.compute_advantages(critic, states, G)
    if verbose :
        print('[train_actor]')
    l_history_actor, _ = tr.fit_actor(actor, actor_optimizer, states, actions, A, n_epochs_actor, verbose)

    return l_history_critic, l_history_actor, rewards.sum().detach()


def train_actor_critic(
    env,
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    device,
    gamma,
    n_episodes=100,
    n_epochs_actor=10,
    n_epochs_critic=10,
    random_reset=True,
    max_steps=200,
    verbose=False,
):
    """
Boucle d’apprentissage Actor–Critic sur plusieurs épisodes.

À chaque épisode :
- une trajectoire est collectée ;
- le critic est entraîné sur les returns ;
- l’acteur est mis à jour à partir des avantages.

Les métriques principales sont enregistrées à chaque épisode.

Parameters
----------
env : MMSimulator
    Environnement de trading simulé.
actor : ActorNet
    Réseau de politique.
critic : CriticNet
    Réseau de valeur.
actor_optimizer : torch.optim.Optimizer
    Optimiseur de l’acteur.
critic_optimizer : torch.optim.Optimizer
    Optimiseur du critic.
device : torch.device
    Device utilisé pour les calculs.
gamma : float
    Facteur d’actualisation.
n_episodes : int
    Nombre total d’épisodes d’entraînement.
n_epochs_actor : int, optional
    Nombre d’itérations d’entraînement de l’acteur par épisode.
n_epochs_critic : int, optional
    Nombre d’itérations d’entraînement du critic par épisode.
random_reset : bool, optional
    Si True, démarre chaque épisode à un point aléatoire.
max_steps : int, optional
    Longueur maximale des épisodes.
verbose : bool, optional
    Affiche des informations pendant l’entraînement.

Returns
-------
history : dict
    Dictionnaire contenant :
    - "critic_loss" : liste des pertes finales du critic par épisode ;
    - "actor_loss" : liste des pertes finales de l’acteur par épisode ;
    - "episode_return" : liste des rewards cumulés par épisode.
""" 
    history = {
    "critic_loss": [],
    "actor_loss": [],
    "episode_return": []
}

    for episode in range(n_episodes):
        l_hist_critic, l_hist_actor, ep_return = train_one_episode(
            env, actor, critic, actor_optimizer, critic_optimizer, device, gamma, n_epochs_actor, n_epochs_critic, random_reset, max_steps, verbose)
        

        history["critic_loss"].append(l_hist_critic[-1])
        history["actor_loss"].append(l_hist_actor[-1])
        history["episode_return"].append(ep_return.cpu())

    return history

def train_one_episode_ppo(env : MMSimulator, actor : ActorNet, critic : CriticNet, 
                      actor_optimizer, critic_optimizer, device, gamma, 
                      n_epochs_actor = 10, n_epochs_critic = 10, random_reset = True, 
                      max_steps = 200, verbose = False):
    """
    Exécute un épisode complet d'apprentissage Actor–Critic avec PPO

    La fonction :
    - collecte une trajectoire sous la politique courante ;
    - calcule les returns à partir des rewards ;
    - entraîne le critic pour approximer la fonction de valeur ;
    - calcule les avantages ;
    - entraîne l’acteur à partir de ces avantages.

    Parameters
    ----------
    env : MMSimulator
        Environnement de trading simulé.
    actor : ActorNet
        Réseau de politique.
    critic : CriticNet
        Réseau de valeur.
    actor_optimizer : torch.optim.Optimizer
        Optimiseur de l’acteur.
    critic_optimizer : torch.optim.Optimizer
        Optimiseur du critic.
    device : torch.device
        Device utilisé pour les calculs.
    gamma : float
        Facteur d’actualisation.
    n_epochs_actor : int, optional
        Nombre d’itérations d’entraînement de l’acteur.
    n_epochs_critic : int, optional
        Nombre d’itérations d’entraînement du critic.
    random_reset : bool, optional
        Si True, démarre l’épisode à un point aléatoire.
    max_steps : int, optional
        Longueur maximale de l’épisode.
    verbose : bool, optional
        Affiche des informations pendant l’entraînement.

    Returns
    -------
    l_history_critic : list
        Historique des pertes du critic sur l’épisode.
    l_history_actor : list
        Historique des pertes de l’acteur sur l’épisode.
    episode_return : float
        Somme des rewards sur l’épisode.
    """
    trajectory = collect_trajectory(env, actor, device, random_reset, max_steps)
    # Récupération
    states = trajectory["states"]
    actions = trajectory["actions"]
    rewards = trajectory["rewards"]
    #pour PPO
    old_log_probs = trajectory["old_log_probs"]

    G = Loss.compute_returns(rewards, gamma)
    if verbose :
        print('[train_critic]')
    l_history_critic, _ = tr.fit_critic(critic, critic_optimizer, states, rewards, gamma, n_epochs_critic, verbose)

    A = Loss.compute_advantages(critic, states, G)
    if verbose :
        print('[train_actor]')
    l_history_actor, _ = tr.fit_actor_ppo(actor, actor_optimizer, states, actions, A,old_log_probs,  n_epochs_actor, verbose)

    return l_history_critic, l_history_actor, rewards.sum().item()




def train_ppo(
    env,
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    device,
    gamma,
    n_episodes=100,
    n_epochs_actor=10,
    n_epochs_critic=10,
    random_reset=True,
    max_steps=200,
    verbose=False,
):
    """
    Boucle d’apprentissage Actor–Critic avec PPO sur plusieurs épisodes.

    À chaque épisode :
    - une trajectoire est collectée ;
    - le critic est entraîné sur les returns ;
    - l’acteur est mis à jour à partir des avantages.

    Les métriques principales sont enregistrées à chaque épisode.

    Parameters
    ----------
    env : MMSimulator
        Environnement de trading simulé.
    actor : ActorNet
        Réseau de politique.
    critic : CriticNet
        Réseau de valeur.
    actor_optimizer : torch.optim.Optimizer
        Optimiseur de l’acteur.
    critic_optimizer : torch.optim.Optimizer
        Optimiseur du critic.
    device : torch.device
        Device utilisé pour les calculs.
    gamma : float
        Facteur d’actualisation.
    n_episodes : int
        Nombre total d’épisodes d’entraînement.
    n_epochs_actor : int, optional
        Nombre d’itérations d’entraînement de l’acteur par épisode.
    n_epochs_critic : int, optional
        Nombre d’itérations d’entraînement du critic par épisode.
    random_reset : bool, optional
        Si True, démarre chaque épisode à un point aléatoire.
    max_steps : int, optional
        Longueur maximale des épisodes.
    verbose : bool, optional
        Affiche des informations pendant l’entraînement.

    Returns
    -------
    history : dict
        Dictionnaire contenant :
        - "critic_loss" : liste des pertes finales du critic par épisode ;
        - "actor_loss" : liste des pertes finales de l’acteur par épisode ;
        - "episode_return" : liste des rewards cumulés par épisode.
    """ 
    history = {
    "critic_loss": [],
    "actor_loss": [],
    "episode_return": []
    }

    for episode in range(n_episodes):
        l_hist_critic, l_hist_actor, ep_return = train_one_episode_ppo(
            env, actor, critic, actor_optimizer, critic_optimizer, device, gamma, n_epochs_actor, n_epochs_critic, random_reset, max_steps, verbose)
        

        history["critic_loss"].append(l_hist_critic[-1])
        history["actor_loss"].append(l_hist_actor[-1])
        history["episode_return"].append(ep_return)

    return history