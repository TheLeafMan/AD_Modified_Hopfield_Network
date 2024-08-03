import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
from PIL import Image

class hopfield:
    def __init__(self, N):
        self.N = N # Number of nodes in the system
        self.current_state = np.zeros(N)
        self.original_state = np.zeros(N)
        self.overall_weight = np.zeros((self.N, self.N))
        
        self.number_of_patterns = 0
        self.pattern_list = []  # Initialize an empty list to store patterns
        self.weight_list = []

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
        
    def asymmetric(self): #Makes hopfield network asymmetrical to make it more "realistic"
        for y in range(self.N):
            for x in range(self.N - y):
                chance = np.random.randint(1,int(self.N*self.N/20))
                # chance = 1
                if chance == 1:
                    temp = np.random.choice([0,1])
                    if temp == 0:
                        self.overall_weight[y][self.N - x - 1] *= 0
                    elif temp == 1:
                        self.overall_weight[self.N - x - 1][y] *= 0
        print("Made asymmetric")
        
    def update_overall_weight(self):
        self.overall_weight = np.zeros((self.N, self.N))  # Reset overall_weight to zeros
        for index in range(self.number_of_patterns):
            self.overall_weight += self.weight_list[index]
        self.overall_weight /= self.number_of_patterns  # Average the values
        self.asymmetric()
        
    def create_pattern(self, i_pattern = None):
        self.number_of_patterns = self.number_of_patterns + 1
        self.create_pattern_matrix(i_pattern)
        self.create_weight_matrix(self.number_of_patterns -1)
        # self.create_potential_matrix(self.number_of_patterns -1)
        self.update_overall_weight()
        
    def update_all_nodes(self):
        nw = np.dot(self.overall_weight, self.current_state)
        self.current_state = np.where(nw > 0, 1, -1)
        # potential = np.matmul(self.overall_weight, self.current_state)
        # for i in range(self.N):
        #     # print(self.current_state[i], end=' ')
        #     if potential[i] > 0:
        #         self.current_state[i] = 1
        #         # print(self.current_state[i])
        #     elif potential[i] < 0:
        #         self.current_state[i] = -1
                # print(self.current_state[i])
        # print("nodes updated")
    
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
        
    def similarity_score(self):
        for i in self.number_of_patterns:
            similarity = self.pattern_list[i]
            similarity = self.current_state - similarity
            unique, counts = np.unique(similarity)
            dict(zip(unique, counts))

        
    def flip_pattern(self):
        # System for flipping if the inverse (FOR DISPLAY ONLY)
        for i in range(self.number_of_patterns):
            if np.all(self.current_state == -1 * self.pattern_list[i]):
                print("Pattern Inverted")
                self.current_state = self.current_state * -1
        
    def image_to_numpy(image_path):
        # Open the image file
        img = Image.open(image_path).convert('L')
        # Convert image to numpy array
        img= np.array(img)
        img_array=np.where(img>30,1,-1)
        return img_array
            
    def kill_neuron(self, node_num):
        self.current_state[node_num] = 0
        for i in range(self.N):
            self.overall_weight[node_num][i] = 0
        for i in range(self.N):
            self.overall_weight[i][node_num] = 0
