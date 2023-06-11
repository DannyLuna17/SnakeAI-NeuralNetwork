SnakeAI-NeuralNetwork
=======
[![CodeFactor](https://www.codefactor.io/repository/github/lunapy17/snakeai-neuralnetwork/badge)](https://www.codefactor.io/repository/github/lunapy17/snakeai-neuralnetwork)

A Python-based project that utilizes deep learning and reinforcement learning to train an AI agent to play the classic Snake game. The project leverages Pygame for the game environment, Keras for the neural network model, and Matplotlib for visualizing training metrics.

# Requirements

* Python 3.x
* Keras
* Matplotlib
* Pygame


# Installation

1. Clone the repository: git clone https://github.com/LunaPy17/SnakeAI-NeuralNetwork
2. Install the required packages: pip install -r requirements.txt

# Usage

To run the application, execute the following command in your terminal:

```
python main.py
```
In the command prompt, you will be given options to either train the model or test the current model. The option to render the game visually during training/testing is also provided.

![example](https://github.com/LunaPy17/SnakeAI-NeuralNetwork/assets/69711934/597efed9-cd91-4f39-b0ec-1920d8a47acc)

# How It Works

The AI agent learns to play the game of Snake using a method called Q-learning, a type of Reinforcement Learning. The agent is rewarded when the snake eats an apple and is penalized when it hits the wall or itself. Over time, the agent learns from its past actions and enhances its performance. The deep learning model, built using Keras, helps the agent decide the best action given a particular game state. This model is trained over numerous episodes, gradually decreasing its tendency to take random actions and instead, base its actions on the predictions of the deep learning model.

# License

This project is licensed under the GPL-3.0 [License](https://github.com/LunaPy17/SnakeAI-NeuralNetwork/blob/main/LICENSE). See the LICENSE file for details.
