"""
algorithmes d'entrainement, mise en oeuvre
"""
from tqdm.auto import tqdm
import torch 
import time
import numpy as np

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

    done = False

    states = []
    next_states = []
    actions = []
    rewards = []
    old_log_probs = []
    dones = []

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action_tensor, log_prob, _ = model.sample_action(state_tensor)          # entropie encore non prise en compte 

        next_state, reward, done = env.step(action_tensor)

        states.append(state_tensor.squeeze(0))          
        actions.append(action_tensor.squeeze(0))        
        old_log_probs.append(log_prob.squeeze(0))       
        rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))
        next_states.append(torch.tensor(next_state, dtype=torch.float32, device=device))
        dones.append(torch.tensor(float(done), dtype=torch.float32, device=device))

        # mise à jour de l'état
        state = next_state

    # Conversion finale en tenseurs
    trajectory = {
        "states": torch.stack(states),
        "actions": torch.stack(actions),
        "rewards": torch.stack(rewards),
        "old_log_probs": torch.stack(old_log_probs),
        "next_states": torch.stack(next_states),
        "dones": torch.stack(dones),

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

def train_one_episode_ppo(
    env,
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    device,
    gamma,
    lam=0.85,
    n_epochs_actor=5,
    n_epochs_critic=10,
    batch_size=32,
    random_reset=True,
    max_steps=200,
    verbose=False
):
    """
    Exécute un épisode complet avec PPO + GAE.
    """
    trajectory = collect_trajectory(
        env=env,
        model=actor,
        device=device,
        reset_rdm=random_reset,
        max_steps=max_steps
    )

    states = trajectory["states"]
    actions = trajectory["actions"]
    rewards = trajectory["rewards"]
    old_log_probs = trajectory["old_log_probs"]
    next_states = trajectory["next_states"]
    dones = trajectory["dones"]

    with torch.no_grad():
        values = critic(states).reshape(-1)
        next_values = critic(next_states).reshape(-1)

    advantages = Loss.compute_gae(
        rewards=rewards,
        values=values,
        next_values=next_values,
        dones=dones,
        gamma=gamma,
        lam=lam
    )

    returns = Loss.compute_ppo_returns(
        advantages=advantages,
        values=values
    )

    if verbose:
        print("[train_critic_ppo]")

    l_history_critic, _ = fit_critic_ppo(
        model=critic,
        optimizer=critic_optimizer,
        states=states,
        returns=returns,
        n_epochs=n_epochs_critic,
        batch_size=batch_size,
        verbose=verbose
    )

    if verbose:
        print("[train_actor_ppo]")

    l_history_actor, _ = fit_actor_ppo(
        model=actor,
        optimizer=actor_optimizer,
        states=states,
        actions=actions,
        old_log_prob=old_log_probs,
        advantages=advantages,
        n_epochs=n_epochs_actor,
        batch_size=batch_size,
        eps_clip=0.25,
        entropy_coef=.01,
        verbose=verbose
    )

    episode_return = rewards.sum().item()

    return l_history_critic, l_history_actor, episode_return

def train_ppo(
    env,
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    device,
    gamma,
    lam=0.85,
    n_episodes=100,
    n_epochs_actor=10,
    n_epochs_critic=20,
    batch_size=256,
    random_reset=True,
    max_steps=200,
    verbose=False,
):
    """
    Boucle PPO sur plusieurs épisodes.
    """
    history = {
        "critic_loss": [],
        "actor_loss": [],
        "episode_return": []
    }
    episode_iter = tqdm(
        range(n_episodes),
        desc="Training PPO",
        disable=not verbose,
        dynamic_ncols=True,
        leave=True,
    )

    for ep in episode_iter:
        t0 = time.time()
        l_hist_critic, l_hist_actor, ep_return = train_one_episode_ppo(
            env=env,
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            device=device,
            gamma=gamma,
            lam=lam,
            n_epochs_actor=n_epochs_actor,
            n_epochs_critic=n_epochs_critic,
            batch_size=batch_size,
            random_reset=random_reset,
            max_steps=max_steps,
            verbose=False
        )
        dt = time.time()-t0
        critic_loss_ep = float(np.mean(l_hist_critic)) if len(l_hist_critic) > 0 else np.nan
        actor_loss_ep = float(np.mean(l_hist_actor)) if len(l_hist_actor) > 0 else np.nan
        
        history["critic_loss"].append(critic_loss_ep)
        history["actor_loss"].append(actor_loss_ep)
        history["episode_return"].append(float(ep_return))
    
    return history
def fit_actor_ppo(
    model: ActorNet,
    optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    n_epochs: int = 10,
    batch_size: int = 256,
    eps_clip: float = 0.25,
    entropy_coef: float = 1.2e-3,
    shuffle: bool = True,
    verbose: bool = False
):
    """
    Boucle d'entraînement PPO de l'acteur avec mini-batches.

    Paramètres
    ----------
    model : ActorNet
    optimizer :
    states : tensor (T, state_dim)
    actions : tensor (T, action_dim)
    old_log_prob : tensor (T,) ou (T,1)
        Log-probabilités sous la politique de collecte.
    advantages : tensor (T,)
        Avantages estimés.
    n_epochs : int
        Nombre d'epochs PPO.
    batch_size : int
        Taille des mini-batches.
    eps_clip : float
        Paramètre de clipping PPO.
    entropy_coef : float
        Coefficient du bonus d'entropie.
    shuffle : bool
        Mélange les indices à chaque epoch.
    verbose : bool
        Affichage de suivi.

    Retourne
    --------
    l_history : list
        Historique moyen des pertes par epoch.
    total_time : float
        Temps d'entraînement.
    """
    l_history = []
    t0 = time.time()
    model.train()

    T = states.shape[0]
    old_log_prob = old_log_prob.detach().reshape(-1)
    advantages = advantages.detach().reshape(-1)

    for epoch in range(n_epochs):
        if shuffle:
            perm = torch.randperm(T, device=states.device)
        else:
            perm = torch.arange(T, device=states.device)

        epoch_losses = []

        for start in range(0, T, batch_size):
            idx = perm[start:start + batch_size]

            batch_states = states[idx]
            batch_actions = actions[idx]
            batch_old_log_prob = old_log_prob[idx]
            batch_advantages = advantages[idx]

            optimizer.zero_grad()

            log_prob, entropy = model.evaluate_actions(batch_states, batch_actions)

            loss = Loss.actor_loss_ppo_fn(
                log_prob=log_prob,
                old_log_prob=batch_old_log_prob,
                A=batch_advantages,
                entropy=entropy,
                eps_clip=eps_clip,
                entropy_coef=entropy_coef
            )

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        l_history.append(mean_loss)

        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            print(
                f"[fit_actor_ppo] epoch={epoch+1}/{n_epochs} "
                f"loss={mean_loss:.4e}"
            )

    elapsed_time = time.time() - t0
    return l_history, elapsed_time


def fit_actor_ppo(
    model,
    optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    n_epochs: int = 10,
    batch_size: int = 256,
    eps_clip: float = 0.25,
    entropy_coef: float = 1.2e-3,
    shuffle: bool = True,
    verbose: bool = False
):
    """
    Boucle d'entraînement PPO de l'acteur avec mini-batches.
    """
    l_history = []
    t0 = time.time()
    model.train()
    T = states.shape[0]
    old_log_prob = old_log_prob.detach().reshape(-1)
    advantages = advantages.detach().reshape(-1)
    for epoch in range(n_epochs):
        if shuffle:
            perm = torch.randperm(T, device=states.device)
        else:
            perm = torch.arange(T, device=states.device)
        epoch_losses = []
        for start in range(0, T, batch_size):
            idx = perm[start:start + batch_size]
            batch_states = states[idx]
            batch_actions = actions[idx]
            batch_old_log_prob = old_log_prob[idx]
            batch_advantages = advantages[idx]
            optimizer.zero_grad()
            log_prob, entropy = model.evaluate_actions(batch_states, batch_actions)
            loss = Loss.actor_loss_ppo_fn(
                log_prob=log_prob,
                old_log_prob=batch_old_log_prob,
                A=batch_advantages,
                entropy=entropy,
                eps_clip=eps_clip,
                entropy_coef=entropy_coef
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())
        mean_loss = float(np.mean(epoch_losses))
        l_history.append(mean_loss)
        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            print(
                f"[fit_actor_ppo] epoch={epoch+1}/{n_epochs} "
                f"loss={mean_loss:.4e}"
            )
    elapsed_time = time.time() - t0
    return l_history, elapsed_time

def fit_critic_ppo(
    model,
    optimizer,
    states: torch.Tensor,
    returns: torch.Tensor,
    n_epochs: int = 20,
    batch_size: int = 256,
    shuffle: bool = True,
    verbose: bool = False
):
    """
    Entraîne le critic sur les returns PPO.
    """
    l_history = []
    t0 = time.time()
    model.train()

    T = states.shape[0]
    returns = returns.detach().reshape(-1)

    for epoch in range(n_epochs):
        if shuffle:
            perm = torch.randperm(T, device=states.device)
        else:
            perm = torch.arange(T, device=states.device)

        epoch_losses = []

        for start in range(0, T, batch_size):
            idx = perm[start:start + batch_size]

            batch_states = states[idx]
            batch_returns = returns[idx]

            optimizer.zero_grad()

            loss = Loss.critic_loss_fn(model, batch_states, batch_returns)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        l_history.append(mean_loss)

        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            print(
                f"[fit_critic_ppo] epoch={epoch+1}/{n_epochs} "
                f"loss={mean_loss:.4e}"
            )

    elapsed_time = time.time() - t0
    return l_history, elapsed_time

