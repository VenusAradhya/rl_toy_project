import gymnasium as gym
import jax
import jax.numpy as jnp
import optax

#1.  environment
env = gym.make("CartPole-v1", render_mode="human")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

#2. policy
def policy_network(params, x):
    x = jnp.tanh(jnp.dot(x, params['W1']) + params['b1'])
    logits = jnp.dot(x, params['W2']) + params['b2']
    return jax.nn.softmax(logits)

#3. parameters
key = jax.random.PRNGKey(42)
key, net_key = jax.random.split(key)

params = {
    'W1': jax.random.normal(net_key, (obs_dim, 16)) * 0.1,
    'b1': jnp.zeros(16),
    'W2': jax.random.normal(net_key, (16, n_actions)) * 0.1,
    'b2': jnp.zeros(n_actions),
}

optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)

#4. loss function
def loss_fn(params, obs, action, reward):
    probs = policy_network(params, obs)
    log_prob = jnp.log(probs[action])
    # We negate because optimizers minimize, but we want to maximize reward
    return -log_prob * reward

#5. JIT Compiled Update Step 
@jax.jit
def update(params, opt_state, obs, action, reward):
    grads = jax.grad(loss_fn)(params, obs, action, reward)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

#6. Training Loop
optimizer = optax.adam(learning_rate=0.001)

for episode in range(1000): # more time to learn
    obs, _ = env.reset()
    #  lists to store episode's data
    states, actions, rewards = [], [], []
    
    done = False
    while not done:
        key, subkey = jax.random.split(key)
        probs = policy_network(params, jnp.array(obs))
        action = int(jax.random.choice(subkey, n_actions, p=probs))
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        # save don't update
        states.append(obs)
        actions.append(action)
        rewards.append(reward)
        
        obs = next_obs
        done = terminated or truncated

    # update after episode
    total_r = sum(rewards)
    for s, a, r in zip(states, actions, rewards):
        params, opt_state = update(params, opt_state, jnp.array(s), a, total_r)

    if episode % 10 == 0:
        print(f"Episode {episode}: Total Reward = {total_r}")

env.close()
