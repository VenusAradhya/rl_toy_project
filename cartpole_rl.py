
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
import numpy as np

#environment
env = gym.make("CartPole-v1", render_mode="human")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n


#policy
def policy(params, x):
    x = jnp.tanh(jnp.dot(x, params['W1']) + params['b1'])
    logits = jnp.dot(x, params['W2']) + params['b2']
    return jax.nn.softmax(logits)

#parameter
key = jax.random.PRNGKey(0)
params = {
    'W1': jax.random.normal(key, (obs_dim, 16)) * 0.1,
    'b1': jnp.zeros(16),
    'W2': jax.random.normal(key, (16, n_actions)) * 0.1,
    'b2': jnp.zeros(n_actions),
}

optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(params)

#action
def select_action(params, obs, key):
    probs = policy(params, obs)
    return jax.random.choice(key, n_actions, p=probs)

#reward
def compute_loss(params, obs, action, reward):
    probs = policy(params, obs)
    log_prob = jnp.log(probs[action])
    return -log_prob * reward

grad_fn = jax.grad(compute_loss)


#training loop
for episode in range(100):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step = 0
    while not done and step < 500:
        key, subkey = jax.random.split(key)
        action = int(select_action(params, jnp.array(obs), subkey))
        obs_next, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # --- Gradient update ---
        grads = grad_fn(params, jnp.array(obs), action, reward)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        obs = obs_next
        step += 1

    print(f"Episode {episode+1}, total reward: {total_reward}")

env.close()
