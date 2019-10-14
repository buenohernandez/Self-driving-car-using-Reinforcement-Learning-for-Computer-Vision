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


from random import randint

clock = pygame.time.Clock()

from collections import deque

import os

path = os.getcwd()

# Deep Q-learning Agent
class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self.load_trained_model()        

    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3,3), activation="relu",  input_shape=self.state_size))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, kernel_size=(3,3), activation="relu",  input_shape=self.state_size))
        model.add(MaxPooling2D(pool_size=(2, 2)))        

        # model.add(BatchNormalization())

        model.add(Flatten())

        model.add(Dense(32, input_dim=self.state_size, activation='relu'))      
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))         
 
        model.add(Dense(self.action_size, activation='linear'))

        return model
        
    def act(self, state):
    
        result = self.model.predict(state)[0].tolist()
        result = result.index(max(result))
        
        return result        
        
    def load_trained_model(self):
       model = self.build_model()
       model.load_weights(path+"/success.model")
       
       return model
       
       
if __name__ == "__main__":

    env = Car()
    agent = DQNAgent(env.state_size,env.act_size)

    trials = 200

    for step in range(trials):
  
        state, _, _ = env.run()

        for trial in range(512):
        
            action = agent.act(state)            
            next_state, reward, done  = env.run(action)    
            state = next_state       
                 
            pressed = pygame.key.get_pressed()            
            if pressed[pygame.K_q]: pygame.quit()            
            
            clock.tick(300)
 

