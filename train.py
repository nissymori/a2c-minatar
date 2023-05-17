import json
import random
from typing import Dict, List, Literal, Union

import git
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from pydantic import BaseModel
from torch.distributions import Categorical

import wandb
from utils import MinAtarEnv


class MinAtarConfig(BaseModel):
    game: Literal[
        "breakout", "asterix", "freeway", "seaquest", "space_invaders"
    ] = "breakout"
    steps: int = int(5e6)
    eval_interval: int = int(1e5)
    eval_n_episodes: int = 64
    eval_deterministic: bool = False
    seed: int = 1234
    num_envs: int = 64
    lr: float = 0.003
    ent_coef: float = 0.0
    gamma: float = 0.99
    value_coef: float = 1.0
    unroll_length: int = 5
    debug: bool = False
    githash: str = git.Repo().head.object.hexsha[:7]
    minatar_version: Literal["v0", "v1"] = "v1"


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


class A2C:
    def __init__(self, config: MinAtarConfig):
        self.config = config

        self.n_steps: int = 0
        self.n_episodes: int = 0
        self.data: Dict[str, List[torch.Tensor]] = {}
        self.env = None
        self.model = None
        self.opt = None

        # stats
        self.n_stats_update = 0
        self.avg_R = 0.0
        self.avg_ent = 0.0
        self.avg_seq_len = 0.0
        self.avg_prob = 0.0
        self.value = 0.0

        self.observations = None

    def train(
        self,
        env,
        model: nn.Module,
        opt,
        n_steps_lim: int = 100_000,
    ) -> Dict[str, float]:
        self.env, self.model, self.opt = env, model, opt

        if self.observations is None:
            self.observations = self.env.reset()  # (num_envs, obs_dim)

        while self.n_steps < n_steps_lim:
            # rollout data
            self.rollout()

            # compute loss and update gradient
            self.opt.zero_grad()
            loss = self.loss()
            loss.backward()
            self.opt.step()

            self.log()

        return {
            "steps": self.n_steps,
            "n_episodes": self.n_episodes,
            f"{self.config.game}/ent": self.avg_ent,
            f"{self.config.game}/prob": self.avg_prob,
            f"{self.config.game}/value": self.value,
            f"{self.config.game}/train_R": self.avg_R,
        }

    def rollout(self) -> None:
        assert self.env is not None and self.model is not None
        self.data = {}
        self.model.train()
        for unroll_ix in range(self.config.unroll_length):
            action, log_prob, entropy, value = self.act(self.observations)  # agent step
            self.observations, rewards, terminated, _ = self.env.step(
                action.numpy()
            )  # env step
            self.n_steps += self.env.num_envs
            truncated = (
                int(unroll_ix == self.config.unroll_length - 1) * (1 - terminated.int())
            ).bool()
            with torch.no_grad():
                _, _, _, next_value = self.act(self.observations)
            self.push_data(
                terminated=terminated,
                truncated=truncated,
                log_prob=log_prob,
                entropy=entropy,
                value=value,
                next_value=next_value,
                rewards=rewards,
            )
            self.observations = self.env.init(terminated)  # auto reset

    def loss(self, reduce=True) -> torch.Tensor:
        v = torch.stack(self.data["value"]).t()  # (num_envs, max_seq_len + 1)
        with torch.no_grad():
            v_tgt = self.compute_return()
        # pg loss
        log_prob = torch.stack(self.data["log_prob"]).t()  # (n_env, seq_len)
        b = v.detach()
        loss = -(v_tgt - b) * log_prob
        # value loss
        value_loss = (v_tgt - v) ** 2
        # ent loss
        ent = torch.stack(self.data["entropy"]).t()  # (num_env, max_seq_len)
        ent_loss = -ent

        loss += self.config.ent_coef * ent_loss
        loss += self.config.value_coef * value_loss
        return loss.sum(dim=1).mean(dim=0) if reduce else loss

    def compute_return(self):
        """compute n-step return following A3C paper"""
        rewards = torch.stack(self.data["rewards"]).t()
        next_values = torch.stack(self.data["next_value"]).t()
        truncated = torch.stack(self.data["truncated"]).t()
        terminated = torch.stack(self.data["terminated"]).t()
        done = truncated | terminated
        R = rewards + self.config.gamma * next_values * truncated.float()
        seq_len = R.size(1)
        for i in reversed(range(seq_len - 1)):
            R[:, i] += self.config.gamma * R[:, i + 1] * (1 - done[:, i].float())
        return R

    def act(self, observations: torch.Tensor):
        assert self.model is not None
        logits, value = self.model(observations)  # (num_envs, action_dim)
        dist = Categorical(logits=logits)
        actions = dist.sample()  # (num_envs)
        log_prob = dist.log_prob(actions)  # (num_envs)
        entropy = dist.entropy()  # (num_envs)
        return actions, log_prob, entropy, value.squeeze()

    def push_data(self, **kwargs) -> None:
        for k, v in kwargs.items():
            assert isinstance(v, torch.Tensor)
            if k not in self.data:
                self.data[k] = []
            self.data[k].append(v)

    def log(self):
        self.n_stats_update += 1

        # logging
        prob = float(torch.exp(torch.stack(self.data["log_prob"])).mean())
        R = float(torch.stack(self.data["rewards"]).sum(dim=0).mean())
        ent = float(torch.stack(self.data["entropy"]).mean())
        v = float(torch.stack(self.data["value"]).mean())

        _avg = lambda x, y, n: (x * n + y * 1) / (n + 1)
        self.avg_R = _avg(self.avg_R, R, self.n_stats_update)
        self.avg_ent = _avg(self.avg_ent, ent, self.n_stats_update)
        self.avg_prob = _avg(self.avg_prob, prob, self.n_stats_update)
        self.value = _avg(self.value, v, self.n_stats_update)


def evaluate(
    env,
    model: nn.Module,
    deterministic: bool = False,
    num_episodes=100,
    time_limit=3000,  # for minatar/seaquest
) -> float:
    model.eval()
    num_envs = env.num_envs
    assert num_episodes % num_envs == 0
    R_seq = []
    for i in range(num_episodes // num_envs):
        obs = env.reset()  # (num_envs, obs_size)
        done = [False for _ in range(num_envs)]
        R = torch.zeros(num_envs)
        t = 0
        while not all(done):
            logits, _ = model(obs)
            dist = Categorical(logits=logits)
            actions = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
            obs, r, done, info = env.step(actions)
            R += r  # If some episode is terminated, all r is zero afterwards.
            t += 1
            if t >= time_limit:
                break
        R_seq.append(R.mean())

    return float(sum(R_seq) / len(R_seq))


args = MinAtarConfig(**OmegaConf.to_object(OmegaConf.from_cli()))
print(args)
wandb.login(key="483ca3866ab4eaa8f523bacae3cb603d27d69c3d")
wandb.init(project=f"a2c-minatar-v1", config=args.dict())


# fix seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


algo = A2C(config=args)
env = MinAtarEnv(game=args.game, num_envs=args.num_envs, seed=args.seed, version=args.minatar_version)
model = ACNetwork(env.num_channels, env.num_actions, args.game)
opt = optim.Adam(model.parameters(), lr=args.lr)

n_train = 0
log = {"steps": 0, f"{args.game}/prob": 1.0 / env.num_actions}
while True:
    log[f"{args.game}/eval_R"] = evaluate(
        MinAtarEnv(
            game=args.game, num_envs=args.num_envs, seed=args.seed + 9999, version=args.minatar_version
        ),  # TODO: fix seed
        model,
        deterministic=args.eval_deterministic,
        num_episodes=args.eval_n_episodes,
    )
    wandb.log(log)
    print(json.dumps(log))
    if algo.n_steps >= args.steps:
        break
    log = algo.train(env, model, opt, n_steps_lim=(n_train + 1) * args.eval_interval)
    torch.save(model.state_dict(), f"params/{args.game}_n_steps_{algo.n_steps}_ent_coef_{args.ent_coef}.pt")
    n_train += 1
