import pgx
import jax 
import jax.numpy as jnp
import numpy as np
import import torch
import troch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from MinAtar import MinAtarEnv

def eval_minatar(model, env, num_episodes, deterministic=False, time_limit=10000):
    model.eval()
    num_envs = env.num_envs
    assert num_episodes % num_envs == 0
    R_seq = []
    A_seq = []
    for i in range(num_episodes // num_envs):
        obs = env.reset()  # (num_envs, obs_size)
        done = [False for _ in range(num_envs)]
        R = torch.zeros(num_envs)
        t = 0
        A = []
        while not all(done):
            logits, _ = model(obs)
            dist = Categorical(logits=logits)
            actions = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
            A.append(actions.numpy())
            obs, r, done, info = env.step(actions)
            R += r  # If some episode is terminated, all r is zero afterwards.
            t += 1
            if t >= time_limit:
                break
        R_seq.append(R.mean())
        A_seq.append(A)

    return R_seq, A_seq


def test_pgx_minatar()

