import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import sys
from PIL import Image

class hopfield:
    def __init__(self, N, dt = 0.1, time = 16):
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
        self.time_steps = int(time/dt)
        self.current_time = 1 # in TIME STEPS
    
        #Tau spread
        self.propagation_threshold = 0.75 # The point by which a neuron will release an NFT to be picked up by another node
        self.propagated_matrix = -1*np.ones(N) #Records how many time steps ago the neuron was propagated -- Will eventually be used in an equation to calculate the dou factor
        self.propagated_matrix[int(self.N/2) + int(np.sqrt(np.sqrt(N))) - 1] = 0 #Seed the very first tau tangle aggregate into the center-most node
        
        self.score_matrix = np.empty((self.time_steps,1))

        self.dou_factor = np.ones((int(self.time_steps), N)) # Factor to be multiplied by weights that represents how damaged each of the synapses connected to the neuron are
        # self.dou_factor[:,int(self.N/2)] = 0   

    #Standard hopfield network functions

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
        
    def create_pattern(self, i_pattern = None):
        self.number_of_patterns = self.number_of_patterns + 1
        self.create_pattern_matrix(i_pattern)
        self.create_weight_matrix(self.number_of_patterns -1)
        # self.create_potential_matrix(self.number_of_patterns -1)
        self.update_overall_weight()
        
    def create_weight_matrix(self, pattern_number):
        weight = np.outer(self.pattern_list[pattern_number], self.pattern_list[pattern_number]) - np.identity(self.N)
        self.weight_list.append(weight)
        
    def update_overall_weight(self):
        self.overall_weight = np.zeros((self.N, self.N))  # Reset overall_weight to zeros
        for index in range(self.number_of_patterns):
            self.overall_weight += self.weight_list[index]
        self.asymmetric()
        self.overall_weight /= self.number_of_patterns  # Average the values
    
    def asymmetric(self): #Makes hopfield network asymmetrical to make it more "realistic"
        for y in range(self.N):
            for x in range(self.N - y):
                chance = np.random.randint(1,10000)
                temp = np.random.choice([0,1])
                if chance == 5:
                    if temp == 0:
                        self.overall_weight[y][self.N - x - 1] *= 0
                    elif temp == 1:
                        self.overall_weight[self.N - x - 1] *= 0
          
    def update_all_nodes(self, variation_of_overall_weight = None): #Implimented system that makes it update by base weight default but you can update it with a damaged matrix as well
        if np.all(variation_of_overall_weight) == None:
            variation_of_overall_weight = self.overall_weight
            
        potential = np.matmul(variation_of_overall_weight, self.current_state)
        for i in range(self.N):
            # print(self.current_state[i], end=' ')
            if potential[i] > 0:
                self.current_state[i] = 1
                # print(self.current_state[i])
            elif potential[i] < 0:
                self.current_state[i] = -1
                # print(self.current_state[i])
        # print("nodes updated")
    
    #Tau functions 
    

    def damage_neurons(self, time_point):
        for i in range(self.N):
            if self.propagated_matrix[i] != -1:
                time_since_propagation = int(time_point - self.propagated_matrix[i])
                new_value = 1 - math.pow((time_since_propagation * self.dt / 6), 2)
                # Ensure value doesn't go below 0
                self.dou_factor[time_point][i] = max(new_value, 0)
        if (((self.dou_factor[time_point][0] == 0) or (self.dou_factor[time_point][0] == -2)) and (time_point < self.time_steps-1)):
            self.dou_factor[time_point + 1][0] = -2 #System in place so that matplotlib does not display all white when everything is dead

    def score_damage(self):
        for time in range(self.time_steps):
            # print(time)
            for i in range(self.N):
                self.score_matrix[time][0] += self.dou_factor[time][i]
            self.score_matrix[time][0] = self.score_matrix[time][0]/self.N
                
    def propagate(self, time_point):
        for i in range(self.N):
            if self.dou_factor[time_point][i] <= self.propagation_threshold:
                nodes_to_propagate = 1
                while nodes_to_propagate != 0:
                    inflicted_node = np.random.random_integers(0,self.N-1)
                    
                    if self.propagated_matrix[inflicted_node] == -1:
                        self.propagated_matrix[inflicted_node] = time_point
                        # print(f"Propagating {inflicted_node}")
                        nodes_to_propagate = 0
                    elif not any(val == -1 for val in self.propagated_matrix):
                        # print(f"All nodes already propagated")
                        nodes_to_propagate = 0
        
    def affect_weight(self, time_point): #Example of usage: myhopfield.update_all_nodes(affect_weight(int(0.25 * myhopfield.time_steps)))
        temp_overall_weight = copy.deepcopy(self.overall_weight)
        for i in range(self.N):
            temp_overall_weight[i] *= self.dou_factor[time_point][i]
        return temp_overall_weight
    
    def tau_process(self):
        while self.current_time < self.time_steps:
            self.damage_neurons(self.current_time)
            self.propagate(self.current_time)
            self.current_time += 1
                # print(self.current_time)
        # print(self.propagated_matrix)
        # print(self.dou_factor)
        self.score_damage()
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        # print(self.score_matrix)
        
        
    # Visualization:
    def create_display(self, ax, arr, title, time_point=None):
        n = len(arr)
        size = math.ceil(math.sqrt(n))

        display_grid = np.full((size, size), -1.0)

        for i in range(n):
            row = i // size
            col = i % size
            display_grid[row, col] = arr[i]
        
        def custom_cmap(val):
            if val == -1:
                return (1., 1., 1., 1.)  # white for padding
            elif val == 1:
                return (0., 0., 0., 0.)  # white
            elif val == 0:
            # elif val < 0.00001:
                return (0., 0., 1., 1.)  # blue for 0

            else:
                # Linear interpolation between white (close to 1) and red (close to 0)
                return (1, val, val, 1)  # redder as val decreases
        
        # Create a color array
        colors = np.array([custom_cmap(val) for val in display_grid.flatten()]).reshape((size, size, 4))
        #print(colors)
        if (((self.dou_factor[time_point][0] == 0) or (self.dou_factor[time_point][0] == -2)) and (time_point < self.time_steps-1)):
            colors[0][0] = (0,0,1,0.9)
        
        ax.imshow(colors, interpolation='nearest')
        ax.set_title(title if time_point is None else f"{title} at {int(time_point*self.dt) + 1} years", fontsize=6)
        print(time_point)
        ax.set_xticks([])
        ax.set_yticks([])

    def visualize_patterns(self, ax, pattern, title):
        dim = int(np.sqrt(self.N))
        pattern_reshaped = pattern.reshape((dim, dim))
        ax.imshow(pattern_reshaped, cmap="gray", vmin=-1, vmax=1)
        ax.set_title(title, fontsize=6)
        ax.axis('off')

    def combined_visualization(self, num_time_steps):
        dim = int(np.sqrt(self.N))
        time_indices = np.linspace(0, self.time_steps - 1, num_time_steps, dtype=int)

        rows = num_time_steps + 1
        cols = self.number_of_patterns  # +2 for Original State and Current State

        fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 1.2))
        fig.tight_layout()

        for i in range(self.number_of_patterns):
            self.visualize_patterns(axs[0, i], self.pattern_list[i], f"Pattern {i + 1}")

        for idx, t in enumerate(time_indices):
            self.create_display(axs[idx + 1, 0], self.dou_factor[t], "Damaged Nodes", t)
            self.set_initial_state()
            self.visualize_patterns(axs[idx + 1, 1], self.original_state, "State Before Update")
            for i in range(5):
                self.update_all_nodes(self.affect_weight(t))
            self.visualize_patterns(axs[idx + 1, 2], self.current_state, "State After Update")

        plt.show()
