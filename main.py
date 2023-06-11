import random
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import logging
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os
import sys

stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
import pygame
sys.stdout = stdout

logging.basicConfig(level=logging.INFO)

# Direction Constants
DIRECTIONS = {'UP': [1, 0, 0, 0], 
              'DOWN': [0, 1, 0, 0], 
              'LEFT': [0, 0, 1, 0], 
              'RIGHT': [0, 0, 0, 1]}

class SnakeGame:
    """
    This class contains the Snake game and its mechanics.
    """

    def __init__(self):
        # Init Game
        pygame.init()
        self.WIDTH, self.HEIGHT = 340, 240
        self.game_window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("SnakeAI")

        # Colors values
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)

        # Snake Attributes
        self.snake_size = 10
        self.snake_speed = 10

        # Apple Status
        self.apple_spawned = True

        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """
        Resets the game state.
        """
        self.snake = [[100, 50], [90, 50], [80, 50]]
        self.snake_direction = 'RIGHT'
        self.apple_position = [random.randrange(1, self.WIDTH//self.snake_size) * self.snake_size, 
                               random.randrange(1, self.HEIGHT//self.snake_size) * self.snake_size]
        return self.get_state()
    
    def render(self):
        """
        Renders the game.
        """
        self.game_window.fill(self.WHITE)

        # Draw The Snake
        for segment in self.snake:
            pygame.draw.rect(self.game_window, self.RED,
                            pygame.Rect(segment[0], segment[1], self.snake_size, self.snake_size))

        # Draw The Apple
        pygame.draw.rect(self.game_window, self.RED,
                        pygame.Rect(self.apple_position[0], self.apple_position[1], self.snake_size, self.snake_size))

        pygame.display.flip()

    def get_state(self):
        """
        Returns the current game state.
        """

        # Snake head position
        snake_head_x = self.snake[0][0] / self.WIDTH
        snake_head_y = self.snake[0][1] / self.HEIGHT

        # Apple position
        apple_x = self.apple_position[0] / self.WIDTH
        apple_y = self.apple_position[1] / self.HEIGHT

        # Snake direction
        direction = DIRECTIONS[self.snake_direction]
        
        return [snake_head_x, snake_head_y, apple_x, apple_y] + direction

    def step(self, action):
        """
        Takes a action in the game.
        """

        # Update the direction of the snake if it is not its current direction 
        if action == 0 and self.snake_direction != 'DOWN':
            self.snake_direction = 'UP'
        elif action == 1 and self.snake_direction != 'UP':
            self.snake_direction = 'DOWN'
        elif action == 2 and self.snake_direction != 'RIGHT':
            self.snake_direction = 'LEFT'
        elif action == 3 and self.snake_direction != 'LEFT':
            self.snake_direction = 'RIGHT'

        # Update the position of the snake
        if self.snake_direction == 'UP':
            self.snake.insert(0, [self.snake[0][0], self.snake[0][1] - self.snake_speed])
        elif self.snake_direction == 'DOWN':
            self.snake.insert(0, [self.snake[0][0], self.snake[0][1] + self.snake_speed])
        elif self.snake_direction == 'LEFT':
            self.snake.insert(0, [self.snake[0][0] - self.snake_speed, self.snake[0][1]])
        else:
            self.snake.insert(0, [self.snake[0][0] + self.snake_speed, self.snake[0][1]])

        # Check if snake hit the wall
        if (self.snake[0][0] >= self.WIDTH or self.snake[0][0] < 0 or 
            self.snake[0][1] >= self.HEIGHT or self.snake[0][1] < 0):
            logging.info("Snake hit the wall. Game Over.")
            return self.get_state(), -10, True
        
        # Check if snake hit itself
        if self.snake[0] in self.snake[1:]:
            logging.info("Snake hit itself. Game Over.")
            return self.get_state(), -10, True
        
        # Check if snake ate the apple
        if self.snake[0] == self.apple_position:
            logging.info("Snake ate an apple.")
            self.apple_position = [random.randrange(1, self.WIDTH//self.snake_size) * self.snake_size,
                                   random.randrange(1, self.HEIGHT//self.snake_size) * self.snake_size]
            return self.get_state(), 10, False
        
        # If none of the above happened, remove the tail of the snake (since it did not eat an apple)
        self.snake.pop()
        return self.get_state(), -1, False
    
class Agent:
    """
    This class represents the RL agent.
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        """
        Constructs the deep learning model.
        """
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """
        Remembers a game step for future learning.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Selects an action for the agent to take given the current game state.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        """
        Trains the agent on remembered game steps.
        """
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """
        Loads the model weights from a file.
        """
        try:
            self.model.load_weights(name)
            print("Successfully loaded existing model weights.")
        except OSError:
            print("No existing model weights found. Training from zero.")

    def save(self, name):
        """
        Saves the model weights to a file.
        """
        self.model.save_weights(name)

def run_episode(game, agent, state_size, render):
    """
    Runs a single episode of the game.
    """

    # Initial game state
    state = game.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    time = 0
    
    # Main game loop
    while not done:
        # Agent makes a move
        action = agent.act(state)

        # Calculate next state and reward
        next_state, reward, done = game.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        
        # Remember the state if not rendering
        if not render:
            agent.remember(state, action, reward, next_state, done)
        else:
            # Rendering the game
            game.render()
            pygame.time.wait(100)

        state = next_state
        time += 1

    return time

def train_model(episodes, render):
    EARLY_STOPPING_SCORE = 500
    game = SnakeGame()
    batch_size = 64
    scores = [] 
    avg_scores = []
    epsilon_values = []

    try:
        agent.load('./model.h5')
    except Exception as e:
        print(f"Error loading weights: {e}")


    for e in range(episodes):
        score = run_episode(game, agent, state_size, render)
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        epsilon_values.append(agent.epsilon)
        print(f"episode: {e}/{episodes}, score: {score}, e: {agent.epsilon:.2f}, average score: {avg_score:.2f}")

        if len(agent.memory) > batch_size and e % 2 == 0:
            agent.replay(batch_size)

        try:
            agent.save('./model.h5')
        except:
            sleep(0.2)
            agent.save('./model.h5')

        if avg_score >= EARLY_STOPPING_SCORE:
            print(f"Early stopping, average score: {avg_score:.2f}")
            break

    plt.plot(scores)
    plt.plot(avg_scores)
    plt.plot(epsilon_values)
    plt.title('Training Metrics')
    plt.ylabel('Value')
    plt.xlabel('Episode')
    plt.legend(['Score', 'Average Score', 'Epsilon'], loc='upper left')
    plt.show()

def testModel(render):
    game = SnakeGame()
    agent.load('./model.h5')
    scores = []

    for e in range(100):
        score = run_episode(game, agent, state_size, render)
        scores.append(score)

    plt.plot(scores)
    plt.title('Test Scores')
    plt.show()

def main():
    """
    Main Function
    """
    print("SnakeAI By https://github.com/LunaPy17\n\n[1] Train Model\n[2] Test Model\n")

    # Take input from the user
    mode = int(input(">>> "))

    render = input("Render Game (Y/N) >>> ").upper() == "Y"

    if mode == 1: return train_model(int(input("Number of Episodes >>> ")), render)
    elif mode == 2: return testModel(render)

state_size = 8
action_size = 4
agent = Agent(state_size, action_size)

if __name__ == "__main__":
    main()