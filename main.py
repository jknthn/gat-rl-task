import random
from collections import deque

import gym
import torch

from src.agent import DeepQNetworkAgent
from src.experience import Experience


def run_deep_q_network_algo(env: gym.Env, agent: DeepQNetworkAgent, episodes: int, eps_start=1.0, eps_end=0.01, eps_decay=0.995, play_only=False):
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    max_step = 1

    for episode_no in range(1, episodes + 1):
        state = env.reset() / 255
        score = 0
        done = False
        step = 0
        while not done:
            if episode_no % 10 == 0 or play_only:
                action = agent.act(state, 0)
            else:
                action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state / 255
            agent.step(Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            ))
            state = next_state
            score += reward
            step += 1
            if episode_no % 10 == 0 or play_only:
                env.render()
            if score < 0:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)
        max_step = max(step, max_step)
        if episode_no % 10 == 0:
            print("#"*80)
            print(f"Episode {episode_no:5d}       score: {score:10.4f}      rolling score: {sum(scores_window) / len(scores_window):10.4f}")
            print("#"*80)
            torch.save(agent.q_network_local.state_dict(), f'models/checkpoint_3_{episode_no}.pth')
        else:
            print(f"Episode {episode_no:5d}       score: {score:10.4f}      eps: {eps:6.4f}")


if __name__ == '__main__':

    SEED = 1

    torch.manual_seed(SEED)
    random.seed(SEED)

    env = gym.make('CarRacing-v0')
    agent = DeepQNetworkAgent(
        state_channels=env.observation_space.shape[-1],
        action_size=env.action_space.shape[0]
    )
    agent.q_network_local.load_state_dict(torch.load("models/checkpoint_2_90.pth", map_location=torch.device('cpu')))
    run_deep_q_network_algo(
        env=env,
        agent=agent,
        episodes=2000,
        eps_start=0.3,
        play_only=True
    )
    env.close()