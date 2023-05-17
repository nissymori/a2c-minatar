import jax
import random
import os
import time

from minatar import Environment

from pgx.minatar import asterix

from minatar_utils import *

state_keys = {
    "player_x",
    "player_y",
    "entities",
    "shot_timer",
    "spawn_speed",
    "spawn_timer",
    "move_speed",
    "move_timer",
    "ramp_timer",
    "ramp_index",
    "terminal",
    "last_action",
}

INF = 99


_spawn_entity = jax.jit(asterix._spawn_entity)
_step_det = jax.jit(asterix._step_det)
_observe = jax.jit(asterix._observe)

def test_spawn_entity():
    entities = jnp.ones((8, 4), dtype=jnp.int32) * INF
    entities = entities.at[:, :].set(
        _spawn_entity(entities, True, True, 1)
    )
    assert entities[1][0] == 0, entities
    assert entities[1][1] == 2, entities
    assert entities[1][2] == 1, entities
    assert entities[1][3] == 1, entities


def test_step_det(model):
    env = Environment("asterix", sticky_action_prob=0.0)
    num_actions = env.num_actions()

    N = 10
    for _ in range(N):
        env.reset()
        done = False
        while not done:
            s = extract_state(env, state_keys)
            a = act(model, env.state())
            r, done = env.act(a)
            lr, is_gold, slot = env.env.lr, env.env.is_gold, env.env.slot
            s_next = extract_state(env, state_keys)
            s_next_pgx = _step_det(
                minatar2pgx(s, asterix.State),
                a,
                lr,
                is_gold,
                slot,
            )
            assert_states(s_next, pgx2minatar(s_next_pgx, state_keys))
            assert r == s_next_pgx.rewards[0]
            assert done == s_next_pgx.terminated


def test_observe(model):
    env = Environment("asterix", sticky_action_prob=0.0)
    num_actions = env.num_actions()

    N = 10
    for _ in range(N):
        env.reset()
        done = False
        while not done:
            s = extract_state(env, state_keys)
            s_pgx = minatar2pgx(s, asterix.State)
            obs_pgx = _observe(s_pgx)
            assert jnp.allclose(
                env.state(),
                obs_pgx,
            )
            a = act(model, env.state())
            r, done = env.act(a)

        # check terminal state
        s = extract_state(env, state_keys)
        s_pgx = minatar2pgx(s, asterix.State)
        obs_pgx = _observe(s_pgx)
        assert jnp.allclose(
            env.state(),
            obs_pgx,
        )


def test_minimal_action_set():
    import pgx
    env = pgx.make("minatar-asterix")
    assert env.num_actions == 5
    state = jax.jit(env.init)(jax.random.PRNGKey(0))
    assert state.legal_action_mask.shape == (5,)
    state = jax.jit(env.step)(state, 0)
    assert state.legal_action_mask.shape == (5,)


if __name__ == "__main__":
    test_minimal_action_set()
    param_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "params")
    env = Environment("asterix", sticky_action_prob=0.0)
    print(f"start testing asterix")
    for filename in os.listdir(param_dir):
        name = filename[:-3]
        if "space_invaders" in name:
            continue
        game, _, _, step_num, _, _, ent_coef = name.split("_")
        if not game == "asterix":
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