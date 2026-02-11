import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
state, _ = env.reset()

done = False
total_reward = 0

while not done:
    env.render()
    action = env.action_space.sample()  #random action
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

print("Total reward:", total_reward)
env.close()


