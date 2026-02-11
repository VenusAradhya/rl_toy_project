import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import gymnasium as gym
import numpy as np

# 1. Define Agent's Brain
class PolicyNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        # The pendulum needs torque between -2 to 2
        return 2.0 * jnp.tanh(x)

# 2. Update Function (Learning logic)
@jax.jit
def train_step(params, opt_state, observations, actions, advantages):
    def loss_fn(params):
        predictions = model.apply(params, observations)
        # a simplified version of Policy Gradient: minimize error
        # between what it is done and what brought good reward 
        return jnp.mean(jnp.square(predictions - actions) * (-advantages))
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss

# 3. Setup initial
env = gym.make("Pendulum-v1", render_mode="rgb_array")
model = PolicyNet()
rng = jax.random.PRNGKey(42)
params = model.init(rng, jnp.zeros((3,)))
optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)

# 4. Training loop
print("Training starts. The pendulum will learn through trial and error...")

for episode in range(3000):
    obs, _ = env.reset()
    episode_data = []
    total_reward = 0
    
    for _ in range(200): # One episode has 200 steps
        # The agent decides the action
        jax_obs = jnp.array(obs)
        # Add noise for exploration
        action = model.apply(params, jax_obs) + np.random.normal(0, 0.2)
        action = np.clip(action, -2.0, 2.0)
        
        # Execution in Gymnasium
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        # Save the data for learning
        episode_data.append((jax_obs, action, reward))
        obs = next_obs
        total_reward += reward
        
        # Show the simulation in real time
        env.render()

    # Process the data and make update at network (Learning)
    obs_batch = jnp.array([d[0] for d in episode_data])
    act_batch = jnp.array([d[1] for d in episode_data])
    rew_batch = jnp.array([d[2] for d in episode_data])
    
    # Normalize the reward (advantage simplified)
    rew_batch = (rew_batch - rew_batch.mean()) / (rew_batch.std() + 1e-5)
    
    params, opt_state, loss = train_step(params, opt_state, obs_batch, act_batch, rew_batch)
    
    if episode % 10 == 0:
        print(f"Episod {episode} | Reward Total: {total_reward:.2f}")

env.close()