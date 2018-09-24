import numpy as np
import random

class ExperienceBuffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
        # print('ExperienceBuffer', 'inited')
    
    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)
            
    def sample(self,size):
        sample = None
        acceptible_size = size
        done = False
        while not done:
            try:
                sample = np.reshape(np.array(random.sample(self.buffer,acceptible_size)),[acceptible_size,5])
                done = True
            except:
                acceptible_size -= 1
                if acceptible_size <= 0:
                    print(size)
                    print("u're fucked. ExpBuff.sample()")
                    exit(0)
        return sample