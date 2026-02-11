import gymnasium as gym

#env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("CartPole-v1", render_mode="human") #wo graphics

for episode in range(3000):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        action = env.action_space.sample()  #random action
        state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        done = terminated or truncated

    if episode %100 == 0:
        print (f"Epsiode {epsiode}: Total = {total_reward}")

print("Total reward:", total_reward)
env.close()




