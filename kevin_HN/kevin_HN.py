import numpy as np
import matplotlib.pyplot as plt
import random
import sys
from PIL import Image
import math

t_a=200
big_cont_count=np.zeros(t_a)
big_exp_count=np.zeros(t_a)

#processing image into np array
class hop:
    #sample inputted
    def __init__(self,init_sample):
        self.mem1=np.array(init_sample[0]).flatten().copy()
        self.mem2=np.array(init_sample[1]).flatten().copy()
        self.states=np.array(init_sample)
        self.dim=self.states.shape
        self.states=np.array(init_sample).flatten()
        self.time=np.zeros(len(self.states))
        self.indices=np.zeros(len(self.states))
        self.tau_count=0
        self.weights=np.zeros((16,16))
        self.time_count=0
        self.cont_count=[]
        self.exp_count=[]
        self.killed_neurons=[]



    #function to update states of each node
    #nstates= neuron states inputted
    #weights= weights between node, to be a NxN array where N is the number of nodes

    def update(self):
        temp_list=[]
        for x in range(10):
            a=self.set_sample(4)
            #a is the memory that it is supposed to converge to

            for y in range(15):
                nw = np.dot(self.weights, self.states)
                new_state = np.where(nw >= 0, 1, -1)
                self.states = new_state
            if a==0:
                temp_list.append(1-compare(self.mem1, self.states))
            else:
                temp_list.append(1-compare(self.mem2, self.states))

        self.cont_count=np.append(self.cont_count,np.mean(temp_list)*100)

    def tau_update(self):
        temp=0
        ###FINISH LATER
        temp_list=[]
        if self.tau_count <16:
            for x in range(math.floor(self.tau_equation(self.time_count))-self.tau_count):
                self.indices[random.randint(0,15)]=1
                self.tau_count=math.floor(self.tau_equation(self.time_count))

        for i in self.indices:
            if i==1:
                self.weaken_neuron(temp,self.time[temp])
                self.time[temp]+=1

            temp+=1

        for x in range(10):
            a=self.set_sample(4)
            #a is the memory that it is supposed to converge to

            for y in range(15):
                nw = np.dot(self.weights, self.states)
                new_state = np.where(nw >= 0, 1, -1)
                self.states = new_state
                for x in self.killed_neurons:
                    self.states[x]=0.5
            if a==0:
                temp_list.append(1-compare(self.mem1, self.states))
            else:
                temp_list.append(1-compare(self.mem2, self.states))

        self.exp_count=np.append(self.exp_count,np.mean(temp_list)*100)

        self.time_count+=1

    #equation to model tau degradation
    def tau_equation(self,time):
        return 1.03**time



    #memory= pattern to memorize
    #must input numpy arrays as memory
    def train_weights(self,mem_list):
        #rets = weights corresponding to memory states
        for pattern in mem_list:
            ret = np.outer(pattern, pattern)    
            np.fill_diagonal(ret, 0)  
            self.weights= self.weights+ret



    def k_connect(self,max):

        x=0
        node_dict={}


        while x != max:
            node1=random.randint(0,len(self.weights[0])-1)
            node2=random.randint(0,len(self.weights)-1) 

            if ((node1,node2) in node_dict.items()) or (node1==node2):
                continue
            node_dict[node1]=[node2]


            self.weights[node1][node2]=0
            x+=1


#160 time steps of one month
    def set_sample(self,max):
        a=random.randint(0,1)
        to_conv=None
        if a ==0:
            to_conv= self.mem1.copy()
        else:
            to_conv= self.mem2.copy()
        x=0
        node_list=[]
        while x != max:
            node=random.randint(0,len(to_conv)-1)

            if node in node_list:

                continue
            node_list.append(node)
            to_conv[node]*=-1
            x+=1
        self.states=to_conv
        return a


    def visualize(self):
        img_array = self.states.reshape(self.dim)
        plt.imshow(img_array, cmap="gray")
        plt.title("Hopfield Network State")
        plt.show()


    def weaken_neuron(self,neuron,time):

        temp=equation(time)
        if temp ==0:
            self.kill_neurons(neuron)
        self.weights[neuron-1,:] =self.weights[neuron-1,:]* temp


    #max= number of neurons killed
    def kill_neurons(self,neuron):

        self.weights[:,neuron-1] = 0
        self.weights[neuron-1,:]=0
        self.killed_neurons.append(neuron-1)


    def kill_rand(self,max):
        x=0
        kill=[]
        while x != max:
            node=random.randint(0,len(self.mem1-1))
            if node in kill:
                continue
            self.kill_neurons(node)
            kill.append(node)
            x+=1

def image_to_numpy(image_path):
    # Open the image file
    img = Image.open(image_path).convert('L')
    # Convert image to numpy array
    img= np.array(img)

    img_array=np.where(img>30,1,-1)

    return img_array
def compare(memory,field):

    return np.mean( memory != field )

def equation(time):
    if time>=60:
        return 0
    return (1-math.pow((time/60),2))
# fig, (ax1,ax2,ax3) = plt.subplots(3) 
def cont_test():
    mem1=np.random.randint(2,size =(4,4))
    mem1=np.where(mem1==0,-1,mem1)
    mem2=np.random.randint(2,size =(4,4))
    mem2=np.where(mem2==0,-1,mem2)

    mem_list=[mem1,mem2]

    field=hop(mem_list)
    field.train_weights(mem_list)
    field.k_connect(20)

    for x in range (t_a):
        field.update()


    return field.cont_count



def exp_test():
    #defining memories
    mem1=np.random.randint(2,size =(4,4))
    mem1=np.where(mem1==0,-1,mem1)
    mem2=np.random.randint(2,size =(4,4))
    mem2=np.where(mem2==0,-1,mem2)

    mem_list=[mem1,mem2]

    field=hop(mem_list)
    field.train_weights(mem_list)
    field.k_connect(20)

    # for x in range (10):
    #     field.update()




    field.kill_rand(3)

    for x in range (t_a):
        field.tau_update()
    return field.exp_count








# memory= [[-1,1,-1,1],
#          [-1,1,1,-1],
#          [1,-1,1,-1],
#          [1,1,1,1]]
# f1=hop(memory)
# f1.train_weights(memory)
# f1.k_connect(256)

# for x in range (3):
#     f1.update()
# # img_array = f1.states.reshape(f1.dim)
# # ax1.imshow(img_array, cmap="gray")

# f1.set_sample(2)
# # img_array = f1.states.reshape(f1.dim)
# # ax2.imshow(img_array, cmap="gray")


# f1.kill_neurons(13)
# f1.kill_neurons(14)
# f1.kill_neurons(15)
# f1.kill_neurons(16)

# for x in range (10):
#     f1.update()
# img_array = f1.states.reshape(f1.dim)
# # ax3.imshow(img_array, cmap="gray")



z=100
for x in range(z):
    big_cont_count=np.add(big_cont_count, cont_test())

for x in range(z):
    big_exp_count=np.add(big_exp_count, exp_test())

big_cont_count=np.divide(big_cont_count,z)
big_exp_count=np.divide(big_exp_count,z)



time=range(len(big_cont_count))
plt.plot(time, big_cont_count)
plt.axis([0, 160, 0, 100])
plt.show()

time=range(len(big_exp_count))
fig2=plt.plot(time, big_exp_count)
plt.axis([0, 160, 0, 100])
plt.show()
