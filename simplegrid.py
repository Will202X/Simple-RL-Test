import numpy as np
import random

grid_width = 3
grid_height = 3

actions = [0, 1, 2, 3]
action_list = [-1, 1, -grid_width, grid_width]

action_space_size = len(action_list)
state_space_size = grid_width * grid_height

goal_square = 8

q_table = np.zeros((state_space_size, action_space_size))

num_episodes = 1000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploratory_decay_rate = 0.001

rewards_all_episodes = []

default_state = 0

# Q-learning algorithm

for episode in range(num_episodes):
    state = default_state

    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:])
        else:
            random_action = random.randint(0, action_space_size - 1)
            action = actions[random_action]

        reward = 0.0

        if state + action_list[action] < state_space_size and \
                state + action_list[action] >= 0:
            new_state = state + action_list[action]
        else:
            new_state = state

        if state == goal_square:
            if step != 0:
                reward = 1.0 - (0.1 * (step // 10))
                reward = round(reward, 1)
            else:
                reward = 1.0
            done = True
        else:
            reward = -1.0

        # Update Q-table for Q(s, a)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        state = new_state
        rewards_current_episode += reward


        if done:
            break

    # Exploration rate decay
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploratory_decay_rate*episode)
    rewards_all_episodes.append(rewards_current_episode)


# Calculate and print the average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1000)
count = 1000
print("Average reward per thousand episodes \n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000

# Print updated Q-table
print("\n \n Q-table \n")
print(q_table)
