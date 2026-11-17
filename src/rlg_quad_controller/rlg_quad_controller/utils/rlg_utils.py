import yaml
import numpy as np
import copy
from types import SimpleNamespace

import torch
import numpy.core.multiarray
from torch.serialization import add_safe_globals



def build_rlg_player(weights_path: str,
                    env_cfg_path: str,
                    agent_cfg_path: str,
                    device: str = None):
    # device=None: usa la GPU se disponibile, altrimenti la CPU
    # rl-games / gym imported here: they are needed only by rl-games (.pth) policies
    from rl_games.torch_runner import Runner
    from rl_games.common import env_configurations
    from gym.spaces import Box

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"


    # --- patch temporanea torch.load con safe globals ---
    add_safe_globals([numpy.core.multiarray.scalar])
    original_torch_load = torch.load

    def patched_torch_load(f, *args, **kwargs):
        kwargs['weights_only'] = False
        # checkpoint salvato su GPU: rimappa sul device scelto (es. CPU)
        kwargs.setdefault('map_location', device)
        return original_torch_load(f, *args, **kwargs)

    torch.load = patched_torch_load

    try:
        with open(agent_cfg_path) as f:
            agent_yaml = yaml.load(f, Loader=yaml.Loader)
        with open(env_cfg_path) as f:
            env_yaml = yaml.load(f, Loader=yaml.Loader)

        params = copy.deepcopy(agent_yaml["params"])
        params["config"]["env_config"] = env_yaml
        params["config"]["device_name"] = device

        obs_dim = 41
        act_dim = 12

        if 'rlgpu' not in env_configurations.configurations:
            dummy_env = SimpleNamespace()
            dummy_env.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
            dummy_env.action_space = Box(
                low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
            )
            env_configurations.configurations['rlgpu'] = {
                'env_creator': lambda **kwargs: dummy_env
            }

        runner = Runner()
        runner.load_config(params=params)
        player = runner.create_player()

        player.restore(weights_path)

        player.model.to(device).eval()
        return player
    finally:
        # Restore torch.load after model is loaded
        torch.load = original_torch_load


class RslRlPlayer:
    """
    Actor di una policy RSL-RL (checkpoint model_*.pt di IsaacLab).
    Espone la stessa interfaccia del player rl-games usata da run_inference
    (attributo .device e metodo get_action), quindi i controller non cambiano.
    """

    ACTIVATIONS = {
        'elu': torch.nn.ELU, 'relu': torch.nn.ReLU, 'selu': torch.nn.SELU,
        'tanh': torch.nn.Tanh, 'sigmoid': torch.nn.Sigmoid, 'lrelu': torch.nn.LeakyReLU,
    }

    def __init__(self, weights_path: str, agent_cfg_path: str, device: str):
        self.device = device

        with open(agent_cfg_path) as f:
            agent_yaml = yaml.load(f, Loader=yaml.Loader)
        actor_cfg = agent_yaml['actor']
        hidden_dims = actor_cfg['hidden_dims']
        activation = self.ACTIVATIONS[actor_cfg['activation']]

        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        if 'actor_state_dict' not in ckpt:
            raise RuntimeError(
                f"{weights_path}: formato RSL-RL non supportato (chiavi: {list(ckpt.keys())})")
        state = ckpt['actor_state_dict']

        # actor ricorrente (RslRlRNNModelCfg): osservazioni normalizzate -> LSTM/GRU -> MLP.
        # Lo stato nascosto resta tra uno step e l'altro (come in rollout in IsaacLab) e si azzera con reset().
        self.rnn = None
        if 'rnn.rnn.weight_ih_l0' in state:
            rnn_type = actor_cfg.get('rnn_type', 'lstm').lower()
            n_layers = int(actor_cfg.get('rnn_num_layers', 1))
            rnn_in, rnn_hidden = state['rnn.rnn.weight_ih_l0'].shape[1], state['rnn.rnn.weight_hh_l0'].shape[1]
            rnn_cls = {'lstm': torch.nn.LSTM, 'gru': torch.nn.GRU}[rnn_type]
            self.rnn = rnn_cls(input_size=rnn_in, hidden_size=rnn_hidden, num_layers=n_layers)
            self.rnn.load_state_dict({k[len('rnn.rnn.'):]: v for k, v in state.items() if k.startswith('rnn.rnn.')})
            self.rnn.to(device).eval()
        self.hidden_state = None

        # testa di stima (EstimatorRNNModel, omniquad lstm_climb_est): stima = MLP(h), la policy legge [h, stima].
        # Struttura dai pesi: Linear, attivazione, ..., Linear (stessa attivazione dell'actor)
        self.estimator = None
        if 'estimator.0.weight' in state:
            n_lin = len([k for k in state if k.startswith('estimator.') and k.endswith('.weight')])
            est_layers = []
            for i in range(n_lin):
                w = state[f'estimator.{2 * i}.weight']
                est_layers.append(torch.nn.Linear(w.shape[1], w.shape[0]))
                if i < n_lin - 1:
                    est_layers.append(activation())
            self.estimator = torch.nn.Sequential(*est_layers)
            self.estimator.load_state_dict({k[len('estimator.'):]: v for k, v in state.items() if k.startswith('estimator.')})
            self.estimator.to(device).eval()
        self.last_estimate = None
        self.estimate_clip = 5.0

        # dimensioni ricavate dai pesi: primo e ultimo layer lineare (con RNN il primo layer riceve lo stato nascosto)
        mlp_in = state['mlp.0.weight'].shape[1]
        obs_dim = self.rnn.input_size if self.rnn is not None else mlp_in
        act_dim = state[f'mlp.{2 * len(hidden_dims)}.weight'].shape[0]

        layers, in_dim = [], mlp_in
        for h in hidden_dims:
            layers += [torch.nn.Linear(in_dim, h), activation()]
            in_dim = h
        layers.append(torch.nn.Linear(in_dim, act_dim))
        self.mlp = torch.nn.Sequential(*layers)
        self.mlp.load_state_dict({k[len('mlp.'):]: v for k, v in state.items() if k.startswith('mlp.')})
        self.mlp.to(device).eval()

        # normalizzazione empirica delle osservazioni (EmpiricalNormalization, eps=1e-2)
        if actor_cfg.get('obs_normalization', False):
            self.obs_mean = state['obs_normalizer._mean'].to(device)
            self.obs_std = state['obs_normalizer._std'].to(device)
        else:
            self.obs_mean, self.obs_std = None, None

        clip = agent_yaml.get('clip_actions')
        self.clip_actions = float(clip) if clip is not None else None
        self.obs_dim, self.act_dim = obs_dim, act_dim

    def reset(self):
        """Azzera lo stato nascosto dell'actor ricorrente (inizio episodio); no-op per gli MLP."""
        self.hidden_state = None

    @torch.no_grad()
    def get_action(self, obs, is_deterministic=True):
        # solo azione media: la policy è valutata in modo deterministico
        if self.obs_mean is not None:
            obs = (obs - self.obs_mean) / (self.obs_std + 1e-2)
        if self.rnn is not None:
            out, self.hidden_state = self.rnn(obs.reshape(1, 1, -1), self.hidden_state)
            obs = out.reshape(1, -1)
            if self.estimator is not None:
                self.last_estimate = self.estimator(obs)
                # EstimatorRNNModel clamps the estimate before the policy MLP reads it (estimate_clip, default 5)
                obs = torch.cat([obs, torch.clamp(self.last_estimate, -self.estimate_clip, self.estimate_clip)], dim=-1)
        action = self.mlp(obs)
        if self.clip_actions is not None:
            action = action.clamp(-self.clip_actions, self.clip_actions)
        return action


def build_player(weights_path: str,
                 env_cfg_path: str,
                 agent_cfg_path: str,
                 device: str = None):
    """Carica una policy rl-games (.pth) o RSL-RL (.pt), riconoscendo il formato dal checkpoint."""
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
    if 'actor_state_dict' in ckpt:
        return RslRlPlayer(weights_path, agent_cfg_path, device)
    return build_rlg_player(weights_path, env_cfg_path, agent_cfg_path, device)




def run_inference(player, observation, det=True):
    obs_torch = torch.from_numpy(observation).to(player.device).float()
    action = player.get_action(obs_torch, is_deterministic=det)
    return action.cpu().numpy()


def run_inference_dict(model, observation):
    with torch.no_grad():
        obs_tensor = torch.from_numpy(observation).to('cuda:0').float()
        return model({
            'is_train':    False,
            'prev_actions': None,
            'obs':         obs_tensor,
            'rnn_states':  None
        })



def run_inference_chat(player, observation, det: bool = True):
    """
    Esegue l'inferenza in modo robusto.
    - det=True: usa la mean della distribuzione (niente sampling ⇒ niente std negativa).
    - det=False: sampling stocastico, con clamp minimo sulla std.
    Accetta observation con shape (obs_dim,) o (1, obs_dim); ritorna (act_dim,).
    """
    # Sanifica e sistemazione shape
    obs = np.asarray(observation, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs[None, ...]  # (1, obs_dim)

    # Evita NaN/Inf che fanno esplodere la sigma
    if not np.isfinite(obs).all():
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

    obs_t = torch.from_numpy(obs).to(player.device)

    with torch.no_grad():
        out = player.model({'obs': obs_t})

        # rl-games di solito restituisce un dict con la distribuzione
        if isinstance(out, dict):
            if 'action_distribution' in out:
                distr = out['action_distribution']
                # Clamp di sicurezza sulla std (se presente come .scale)
                if hasattr(distr, 'scale'):
                    distr.scale.clamp_(min=1e-6)
                action_t = distr.mean if det else distr.sample()
            elif 'action' in out:  # fallback: alcuni modelli danno già l'azione
                action_t = out['action']
            else:
                # ultimo fallback: se il dict ha altro formato
                raise RuntimeError("Output del model non riconosciuto: manca 'action_distribution'/'action'")
        else:
            # Se il modello restituisce direttamente un tensore azione
            if torch.is_tensor(out):
                action_t = out
            else:
                raise RuntimeError("Output del model non riconosciuto")

    # shape finale (act_dim,)
    return action_t.squeeze(0).detach().cpu().numpy()
