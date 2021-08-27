# GAT Reinforcement Learning Task

## 1. Environment

I've chosen `CarRace-v0` due to having solved `FrozenLake` in the past.

## 2. Approach

With `FrozenLake` environment simpler approach of tabular reinforcement learning like Q-Learning or Monte Carlo based would work well due to limited state space (8x8) positions.

`CarRace` state is image sized 96x96x3 with values ranging from 0 to 255 per pixel rendering classic reinforcement learning unusable. I've decided to use my own implementation of [Deep Q Network](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) and adapt it to `CarRace` problem. 

To optimize training and experience gathering I've decided to break episode after reaching negative cumulative reward. For validation I also run agent every 10 episodes without exploration (`eps`).

As final approach I've started to discretize action space into 12 presets.

## 3. Results

Hard to say about results as training took time and I didn't finish it properly. With provided model `checkpoint_4_160.pth` the car can "sometimes" turn at first turn and you can see it "tries" to stay in lane.

## 4. Next steps

1. Get access to GPU machine (Colab run out of RAM)
2. Train for some serious no of episodes
3. Code refactor, create run and train scripts
4. Dockerize
5. DQN improvements [RAINBOW](https://arxiv.org/abs/1710.02298)


## 5. Conclusions

I should've solve `FrozenLake`
