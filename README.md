# Snake Game Using PPO Reinforcement Learning

This project implements a **Snake Game** using **Proximal Policy Optimization (PPO)**, a state-of-the-art Reinforcement Learning algorithm. The aim is to build an AI agent that learns how to play the classic Snake game efficiently by training on its environment using PPO.

## Key Features

- **PPO Reinforcement Learning:** The project uses the PPO algorithm, an on-policy RL method, to train the snake agent on how to collect food, avoid walls, and avoid its own tail.
- **Game Environment:** The snake game environment is built using Python, allowing the agent to interact with the game, receive feedback (reward/punishment), and take actions based on its state.
- **Training and Testing:** The agent is trained through multiple episodes and evaluated based on performance metrics such as the number of food items collected and survival time.
- **Visualization:** The project includes graphical visualization of the game, where you can observe the agent playing in real-time.

## Technologies Used

- **Programming Language:** Python
- **Reinforcement Learning Framework:** Stable-Baselines3 (PPO implementation)
- **Libraries:** NumPy, Matplotlib, Pygame (for game environment visualization)
- **Development Environment:** Jupyter Notebook / Python

## Project Structure

The repository contains the following files and directories:

- `snake_game.py`: Contains the code for the Snake game environment.
- `ppo_agent.py`: Code for setting up the PPO agent, training it, and evaluating its performance.
- `train.py`: Script to train the PPO agent on the Snake game environment.
- `requirements.txt`: A list of required Python packages.
- `logs/`: Directory to store training logs and models.

## Installation Instructions

Follow the steps below to get started with this project:

### 1. Clone the repository

```bash
git clone https://github.com/Vejandlachakrish/Snake-Game-using-PPO-RL.git
```
2. Navigate to the project directory
```bash
cd Snake-Game-using-PPO-RL
```
3. Install the required dependencies
Before running the application, install the required Python dependencies by running:
```bash
pip install -r requirements.txt
```
4. Run the training script
To train the PPO agent, run:

```bash
python train.py
```
5. Visualize the agent playing the game
Once the agent is trained, you can visualize the agent playing the game by running the following:

```bash
python snake_game.py
```
## How It Works
Game Environment: The Snake game environment is built where the agent controls the snake and tries to eat the food while avoiding obstacles. The game is implemented using the Pygame library.

Reinforcement Learning: The PPO algorithm trains the agent using the environment's feedback. The agent receives a reward for eating food and a penalty for hitting obstacles or the snake’s own body.

Training: The PPO agent is trained on the environment, where it learns the optimal policy to navigate the snake to maximize its score.

Visualization: The game and agent’s performance are visualized through graphical output to show the real-time learning process.

## Evaluation Metrics
Total Score: The total score is measured by the number of food items the snake consumes before hitting an obstacle or itself.

Survival Time: The total time the snake survives in the game before colliding with a wall or its own body.

## Contributing
If you would like to contribute to this project, feel free to fork the repository, create a new branch, and submit a pull request with your proposed changes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
