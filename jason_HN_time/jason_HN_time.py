import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
from PIL import Image

class hopfield:
    def __init__(self, N, dt = 1, time = 16):
        #Basic Hopfield Network parameters
        self.N = N # Number of nodes in the system
        self.current_state = np.zeros(N)
        self.original_state = np.zeros(N)
        self.overall_weight = np.zeros((self.N, self.N))

        self.number_of_patterns = 0
        self.pattern_list = []  # Initialize an empty list to store patterns
        self.weight_list = []
        
        #Time parameters
        self.time = time # years
        self.dt = dt # portion of year each time step is
        self.time_steps = time/dt
        self.current_time = 0
    
        #Tau spread
        self.propagated = -1*np.ones(N) #Records how many time steps ago the neuron was propagated -- Will eventually be used in an equation to calculate the dou factor
        self.propagated[int(self.N/2)] = 0 #Seed the very first tau tangle aggregate into the center-most node
        self.dou_factor = np.ones((int(self.time_steps), N)) # Factor to be multiplied by weights that represents how damaged each of the synapses connected to the neuron are
        self.dou_factor[:,int(self.N/2)] = 0
        

    def create_pattern_matrix(self, inputted_pattern = None): #Still needs to implement system that raises value error if the pattern is a different size
        if np.all(inputted_pattern == None):
        #Creates a pattern through random generation
            pattern = np.random.choice([1, -1], size=self.N)
        else:
            inputted_pattern = inputted_pattern.flatten()
            pattern = inputted_pattern
        self.pattern_list.append(pattern)  # Store the new pattern in the list
        
    def set_initial_state(self, inputted_pattern = None):
        if np.all(inputted_pattern == None):
        #Creates a pattern through random generation
            pattern = np.random.choice([1, -1], size=self.N)
            self.current_state = pattern
            self.original_state = pattern
        else:
            self.current_state = copy.deepcopy(inputted_pattern)
            self.original_state = copy.deepcopy(inputted_pattern)
        
    def create_weight_matrix(self, pattern_number):
        weight = np.outer(self.pattern_list[pattern_number], self.pattern_list[pattern_number]) - np.identity(self.N)
        self.weight_list.append(weight)
        
    def update_overall_weight(self):
        self.overall_weight = np.zeros((self.N, self.N))  # Reset overall_weight to zeros
        for index in range(self.number_of_patterns):
            self.overall_weight += self.weight_list[index]
        self.asymetric()
        self.overall_weight /= self.number_of_patterns  # Average the values
    
    def asymmetric(self):
        for y in range(self.N):
            for x in range(self.N - y):
                temp = np.random.choice([0,1])
                if temp == 0:
                    self.overall_weight[y][self.N - x] *= 0
                elif temp == 1:
                    self.overall_weight[self.N - x] *= 0
    
    def create_pattern(self, i_pattern = None):
        self.number_of_patterns = self.number_of_patterns + 1
        self.create_pattern_matrix(i_pattern)
        self.create_weight_matrix(self.number_of_patterns -1)
        # self.create_potential_matrix(self.number_of_patterns -1)
        self.update_overall_weight()
        
    def update_all_nodes(self):
        # nw = np.dot(self.overall_weight, self.current_state)
        # self.current_state = np.where(nw >= 0, 1, -1)
        potential = np.matmul(self.overall_weight, self.current_state)
        for i in range(self.N):
            # print(self.current_state[i], end=' ')
            if potential[i] > 0:
                self.current_state[i] = 1
                # print(self.current_state[i])
            elif potential[i] < 0:
                self.current_state[i] = -1
                print(self.current_state[i])
        print("nodes updated")
    
    def visualize_patterns(self, pattern):
        display_patterns = self.pattern_list
        display_patterns.append(self.original_state)
        display_patterns.append(self.current_state)
        
        dim = int(np.sqrt(self.N))  # Assuming the original pattern is square-shaped
        fig, axs = plt.subplots(1, self.number_of_patterns + 2, figsize=(self.number_of_patterns * 5, 5))
        if self.number_of_patterns == 1:
            axs = [axs]  # Ensure axs is always a list even if there's only one pattern

        for i, pattern in enumerate(display_patterns):
            pattern_reshaped = pattern.reshape((dim, dim))
            axs[i].imshow(pattern_reshaped, cmap="gray", vmin=-1, vmax=1)
            if i < self.number_of_patterns:
                axs[i].set_title(f"Pattern {i + 1}")
            elif i == self.number_of_patterns:
                axs[i].set_title("Original State")
            elif i == self.number_of_patterns + 1:
                axs[i].set_title("Current_State")
            axs[i].axis('off')   
        plt.show()

    #Tau functions
    # def damage_singular_neuron(self, neuron_damaged):
        ## After 1 year, the neuron is damaged by 50%
        # dd_dt = 0.5
        # self.propagated[self.current_time + 1][neuron_damaged] = self.propagated[self.current_time][neuron_damaged] - dd_dt * self.dt #Using forward euler method

    # def damage_neurons(self):
    #     pass
        

    def propagate(self, time_point):
        for i in range(self.N):
            if self.dou_factor[time_point][i] < 0.5:
                self.propagated[i + 1] = self.current_time + 1
                
    # def tau_process(self):
    #     for time in range(self.time_steps):
    #         if(self.current_time <= self.time_steps):
    #             self.damage_neurons()
    #             self.propagate(time)
    #             self.current_time += 1