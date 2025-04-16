import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import PPO_Net
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
PPO_EPOCHS = 10
MINI_BATCH_SIZE = 64
CLIP_EPS = 0.2

class Agent:
    def __init__(self):
        self.n_games = 0
        self.gamma = 0.9
        self.model = PPO_Net(14, 256, 3)  # Updated input_size from 11 to 14
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.rollout = []
        self.action_size = 3

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight (wall or self)
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right (wall or self)
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left (wall or self)
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Obstacle straight
            (dir_r and point_r in game.obstacles) or
            (dir_l and point_l in game.obstacles) or
            (dir_u and point_u in game.obstacles) or
            (dir_d and point_d in game.obstacles),

            # Obstacle right
            (dir_u and point_r in game.obstacles) or
            (dir_d and point_l in game.obstacles) or
            (dir_l and point_u in game.obstacles) or
            (dir_r and point_d in game.obstacles),

            # Obstacle left
            (dir_d and point_r in game.obstacles) or
            (dir_u and point_l in game.obstacles) or
            (dir_r and point_u in game.obstacles) or
            (dir_l and point_d in game.obstacles),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, done, log_prob, value):
        self.rollout.append((state, action, reward, done, log_prob.item(), value.item()))

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float)
        policy_logits, value = self.model(state)
        probs = F.softmax(policy_logits, dim=0)
        action = torch.multinomial(probs, 1).item()
        log_prob = F.log_softmax(policy_logits, dim=0)[action]
        final_move = [0] * self.action_size
        final_move[action] = 1
        return final_move, action, log_prob, value

    def train_ppo(self, next_value):
        if len(self.rollout) < BATCH_SIZE:
            return

        returns = []
        ret = next_value if not self.rollout[-1][3] else 0
        for t in reversed(range(len(self.rollout))):
            reward, done = self.rollout[t][2], self.rollout[t][3]
            ret = reward + self.gamma * ret * (1 - done)
            returns.insert(0, ret)

        states = torch.tensor([item[0] for item in self.rollout], dtype=torch.float)
        actions = torch.tensor([item[1] for item in self.rollout], dtype=torch.long)
        log_probs_old = torch.tensor([item[4] for item in self.rollout], dtype=torch.float)
        values_old = torch.tensor([item[5] for item in self.rollout], dtype=torch.float)
        returns = torch.tensor(returns, dtype=torch.float)
        advantages = returns - values_old
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(PPO_EPOCHS):
            indices = np.random.permutation(len(self.rollout))
            for start in range(0, len(self.rollout), MINI_BATCH_SIZE):
                end = start + MINI_BATCH_SIZE
                idx = indices[start:end]
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_log_probs_old = log_probs_old[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]

                policy_logits, values = self.model(batch_states)
                probs = F.softmax(policy_logits, dim=1)
                log_probs = F.log_softmax(policy_logits, dim=1)
                batch_log_probs = log_probs.gather(1, batch_actions.unsqueeze(1)).squeeze()

                ratio = torch.exp(batch_log_probs - batch_log_probs_old)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                loss = policy_loss + 0.5 * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.rollout.clear()

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        state_old = agent.get_state(game)
        final_move, action, log_prob, value = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        agent.remember(state_old, action, reward, done, log_prob, value)
        if len(agent.rollout) >= BATCH_SIZE:
            if not done:
                _, next_value = agent.model(torch.tensor(state_new, dtype=torch.float))
                next_value = next_value.item()
            else:
                next_value = 0
            agent.train_ppo(next_value)
        if done:
            game.reset()
            agent.n_games += 1
            if score > record:
                record = score
                agent.model.save()
            print('Game', agent.n_games, 'Score', score, 'Record:', record)
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()