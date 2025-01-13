import numpy as np
from collections import deque
import random

class Memory():
    def __init__(self,members, groups):
        self.memory = {}

        # temp memory to store last-step state and action because of no immediate feedback
        self.temp_memory = {}
        self.experience = 0
        for m in members:
            self.temp_memory[m]={'s':[],'a':[],'fp':[],'r':[] }
        for g in groups:
            self.memory[g] = deque(maxlen=2000)




    def remember(self, state,fp, action, reward, next_state,next_fp,member_id):
        self.experience+=1
        self.memory[member_id].append((state,fp, action, reward, next_state,next_fp ))