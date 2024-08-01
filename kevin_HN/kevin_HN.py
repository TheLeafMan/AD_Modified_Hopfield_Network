import numpy as np
import matplotlib.pyplot as plt
import random
import sys
from PIL import Image
np.set_printoptions(threshold=sys.maxsize)
#processing image into np array
class hop:
    #sample inputted
    def __init__(self,init_sample):
        self.states=np.array(init_sample)
        self.dim=self.states.shape
        self.states=np.array(init_sample).flatten()
    #function to update states of each node
    #nstates= neuron states inputted
    #weights= weights between node, to be a NxN array where N is the number of nodes
    def update(self):
        #holder for new state
        new_state= np.array(self.states)
        for i in range(len(self.states)):
            #new weight
            nw=0

            for j in range(len(self.states)):
                nw+=int(self.states[j])* int(self.weights[i][j])

            if nw>=0:
                new_state[i]=1
            else:
                new_state[i]=-1
        #storing states in object

        self.states=new_state

    #memory= pattern to memorize
    def train_weights(self,memory):
        #rets = weights corresponding to memory states
        ret = np.outer(memory, memory)
        for i in range(len(memory)*len(memory[0])):
            ret[i][i]=0
            #storing weights into class
        print(ret)
        self.weights=ret

    def visualize(self):
        img_array = self.states.reshape(self.dim)
        plt.imshow(img_array, cmap="gray")
        plt.title("Hopfield Network State")
        plt.show()
    #max= number of neurons killed
    def kill_neurons(self,max):

        for x in range(max):
            r=random.randint(0,len(self.weights)-1)
            print(r)
            for i in range(len(self.states)):
                for x in range(len(self.weights[r])):
                    self.weights[r][x]=0
            for i in range(len(self.states)):
                    self.weights[i][r]=0


sample=[[-1,1,-1,-1,1,1],[-1,1,1,1,1,-1],
     [-1,1,1,-1,1,-1],[-1,1,-1,1,-1,1],
     [-1,1,-1,1,-1,1],[-1,1,1,1,1,-1]
     ]

real=[[-1,1,-1,-1,1,-1,],
      [-1,-1,1,-1,1,-1],
      [-1,1,-1,-1,-1,1],[-1,1,-1,-1,1,-1,],
      [-1,-1,1,1,-1,1],[-1,1,1,1,1,-1]
      ]


fig, axs = plt.subplots(3) 
f1=hop(sample)
f2=hop(sample)
f1.train_weights(real)
f1.kill_neurons(10)

for i in range(10):
    f1.update()

img_array = f1.states.reshape(f1.dim)
axs[0].imshow(img_array, cmap="gray")

f2.train_weights(real)

for i in range(10):
    f2.update()

img_array = f2.states.reshape(f2.dim)

axs[1].imshow(img_array, cmap="gray")
img_array = np.array(real).reshape(f2.dim)
axs[2].imshow(img_array, cmap="gray")
plt.show()