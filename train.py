from game_cart import Car
import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD, Adadelta, RMSprop
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten
import cv2
from keras.layers.normalization import BatchNormalization
import pygame
import numpy as np
from random import randint
from time import time, sleep
from collections import deque

class DQNAgent:

    
    def __init__(self, state_size, action_size):
        
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.9    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

        
    def _build_model(self):

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3,3), activation="relu",  input_shape=self.state_size))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, kernel_size=(3,3), activation="relu",  input_shape=self.state_size))
        model.add(MaxPooling2D(pool_size=(2, 2)))     
        model.add(Flatten())
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))      
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))         
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',optimizer=SGD(lr=self.learning_rate))
        
        return model

    
    def remember(self, state, action, reward, new_state, done):
        
        self.memory.append([state, action, reward, new_state, done])

        
    def act(self, state):
        
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        
        if np.random.random() < self.epsilon:
            
            return random.randrange(self.action_size)
        
        return np.argmax(self.model.predict(state)[0])

    
    def replay(self, batch_size):

        if len(self.memory) < batch_size:
            
            return

        samples = random.sample(self.memory, batch_size)

        for sample in samples:
            
            state, action, reward, new_state, done = sample
            target = self.model.predict(state)
            
            if done:
                
                target[0][action] = reward
                
            else:
                
                Q_future = max(self.model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
                
            self.model.fit(state, target, epochs=1, verbose=0)

    def save_model(self, fn):

        self.model.save(fn)



if __name__ == "__main__":

    env = Car()
    agent = DQNAgent(env.state_size,env.act_size)
    start = time()
    trials = 100 
    run = 0
    max_run = 0

    for step in range(trials):
        
        state, _, _ = env.run()

        for trial in range(512):

            action = agent.act(state)
            next_state, reward, done  = env.run(action)  
            
            if reward != -10: 
                
                run += 1
                
                if run > max_run: max_run = run
                
            else:
                run = 0
                            
            print(step, trial, reward, action, run, max_run)
            agent.remember(state, action, reward, next_state, done)
            agent.replay(32)

            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_q]: pygame.quit()

            state = next_state

    agent.save_model("success.model")




