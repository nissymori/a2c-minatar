import random
import jax
import os
import time
from minatar import Environment

from pgx.minatar import breakout

from minatar_utils import *

state_keys = {
    "ball_y",
    "ball_x",
    "ball_dir",
    "pos",
    "brick_map",
    "strike",
    "last_x",
    "last_y",
    "terminal",
    "last_action",
}
_step_det = jax.jit(breakout._step_det)
_init_det = jax.jit(breakout._init_det)
observe = jax.jit(breakout._observe)

def test_step_det(model):
    env = Environment("breakout", sticky_action_prob=0.0)
    num_actions = env.num_actions()

    N = 10
    for n in range(N):
        env.reset()
        done = False
        while not done:
            s = extract_state(env, state_keys)
            a = act(model, env.state())
            r, done = env.act(a)
            s_next = extract_state(env, state_keys)
            s_next_pgx = _step_det(
                minatar2pgx(s, breakout.State), a
            )
            assert_states(s_next, pgx2minatar(s_next_pgx, state_keys))
            assert r == s_next_pgx.rewards[0]
            assert done == s_next_pgx.terminated


def test_init_det():
    env = Environment("breakout", sticky_action_prob=0.0)
    N = 1
    for _ in range(N):
        env.reset()
        ball_start = 0 if env.env.ball_x == 0 else 1
        s = extract_state(env, state_keys)
        s_pgx = _init_det(ball_start)
        assert_states(s, pgx2minatar(s_pgx, state_keys))


def test_observe(model):
    env = Environment("breakout", sticky_action_prob=0.0)
    num_actions = env.num_actions()

    N = 10
    for _ in range(N):
        env.reset()
        done = False
        while not done:
            s = extract_state(env, state_keys)
            s_pgx = minatar2pgx(s, breakout.State)
            obs_pgx = observe(s_pgx)
            assert jnp.allclose(
                env.state(),
                obs_pgx,
            )
            a = act(model, env.state())
            r, done = env.act(a)

        # check terminal state
        s = extract_state(env, state_keys)
        s_pgx = minatar2pgx(s, breakout.State)
        obs_pgx = observe(s_pgx)
        assert jnp.allclose(
            env.state(),
            obs_pgx,
        )


def test_minimal_action_set():
    import pgx
    env = pgx.make("minatar-breakout")
    assert env.num_actions == 3
    state = jax.jit(env.init)(jax.random.PRNGKey(0))
    assert state.legal_action_mask.shape == (3,)
    state = jax.jit(env.step)(state, 0)
    assert state.legal_action_mask.shape == (3,)


if __name__ == "__main__":
    test_init_det()
    test_minimal_action_set()
    param_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "params")
    env = Environment("breakout", sticky_action_prob=0.0)
    print(f"start testing breakout")
    for filename in os.listdir(param_dir):
        name = filename[:-3]
        if "space_invaders" in name:
            continue
        game, _, _, step_num, _, _, ent_coef = name.split("_")
        if not game == "breakout":
            continue
        print(f"start testing with model n_steps{str(step_num)} ent_coef{str(ent_coef)}")
        sta = time.time()
        filepath = os.path.join(param_dir, filename)
        model = load_model(filepath, env, game)
        test_step_det(model)
        test_observe(model)
        end = time.time()
        print(f"finish testing with model n_steps{str(step_num)} ent_coef{str(ent_coef)} in {end - sta} seconds")
    print("all tests passed")