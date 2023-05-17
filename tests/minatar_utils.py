import copy
from typing import Any, Dict

import numpy as np
from jax import numpy as jnp

import torch
import torch.nn as nn
from torch.distributions import Categorical


INF = 99

def extract_state(env, state_keys):
    state_dict = {}
    # task-dependent attribute
    for k in state_keys:
        state_dict[k] = copy.deepcopy(getattr(env.env, k))
    return state_dict


def assert_states(state1, state2):
    keys = state1.keys()
    assert keys == state2.keys()
    for key in keys:
        if key == "entities":
            assert len(state1[key]) == len(state2[key])
            for s1, s2 in zip(state1[key], state2[key]):
                assert s1 == s2, f"{s1}, {s2}\n{state1}\n{state2}"
        else:
            assert np.allclose(
                state1[key], state2[key]
            ), f"{key}, {state1[key]}, {state2[key]}\n{state1}\n{state2}"


def pgx2minatar(state, keys) -> Dict[str, Any]:
    d = {}
    for key in keys:
        d[key] = copy.deepcopy(getattr(state, "_" + key))
        if isinstance(d[key], jnp.ndarray):
            d[key] = np.array(d[key])
        if key == "entities":
            val = [None] * 8
            for i in range(8):
                if d[key][i][0] != INF:
                    e = [d[key][i][j] for j in range(4)]
                    val[i] = e
            d[key] = val
    return d


def minatar2pgx(state_dict: Dict[str, Any], state_cls):
    d = {}
    for key in state_dict.keys():
        val = copy.deepcopy(state_dict[key])

        # Exception in Asterix
        if key == "entities":
            _val = [[INF if x is None else x[j] for j in range(4)] for i, x in enumerate(val)]
            val = jnp.array(_val, dtype=jnp.int32)
            d[key] = val
            continue

        # Exception in Seaquest
        if key in ["f_bullets", "e_bullets", "e_fish", "e_subs", "divers"]:
            N = 25 if key.startswith("e_") else 5
            M = 3 if key.endswith("bullets") else 4
            if key == "e_subs":
                M = 5
            v = - jnp.ones((N, M), dtype=jnp.int32)
            for i, x in enumerate(val):
                v = v.at[i, :].set(jnp.array(x))
            d[key] = v
            continue

        # Cast to int32
        if key in ["terminate_timer", "oxygen"]:
            val = jnp.array(val, dtype=jnp.int32)
            d[key] = val
            continue

        # Cast to bool
        if isinstance(val, np.ndarray):
            if key in (
                "brick_map",
                "alien_map",
                "f_bullet_map",
                "e_bullet_map",
                "allien_map",
            ):
                val = jnp.array(val, dtype=jnp.bool_)
            else:
                val = jnp.array(val, dtype=jnp.int32)
            d[key] = val
            continue

        if key in ["terminal", "sub_or", "surface"]:
            val = jnp.array(val, dtype=jnp.bool_)
        else:
            val = jnp.array(val, dtype=jnp.int32)
        d[key] = val

    d = {"_" + k: v for k, v in d.items()}
    s = state_cls(**d)
    return s


class ACNetwork(nn.Module):
    """Modified from MinAtar example:
    - https://github.com/kenjyoung/MinAtar/blob/master/examples/AC_lambda.py
    """

    def __init__(self, in_channels, num_actions, env_name):
        super(ACNetwork, self).__init__()

        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)

        self.policy = nn.Linear(in_features=128, out_features=num_actions)
        self.value = nn.Linear(in_features=128, out_features=1)
        nn.init.constant_(self.value.bias, 0.0)
        nn.init.constant_(self.value.weight, 0.0)

    # As per implementation instructions, the forward function should be overwritten by all subclasses
    def forward(self, x):
        dSiLU = lambda x: torch.sigmoid(x) * (1 + x * (1 - torch.sigmoid(x)))
        SiLU = lambda x: x * torch.sigmoid(x)

        x = x.reshape((x.shape[0], -1, 10, 10))  # (n_samples, channels in env, 10, 10)
        x = SiLU(self.conv(x))
        x = dSiLU(self.fc_hidden(x.view(x.size(0), -1)))
        return self.policy(x), self.value(x)


def load_model(model_file_path, minarar_env):
    model = ACNetwork(env.num_channels, env.num_actions, args.game)
    model.load_state_dict(torch.load(model_file_path))
    model.eval()
    return model


def act(model, minatar_state, deterministic=False):
    obs = minatar_state.state()
    obs = torch.from_numpy(obs).float().unsqueeze(0)
    probs, _ = model(obs)
    m = Categorical(probs)
    action = m.sample() if not deterministic else m.probs.argmax()
    return action.item()

