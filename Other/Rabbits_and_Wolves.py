import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.random import RandomState


def compute_surroundings(ij, N, M):
    max_n = N - 1
    max_m = M - 1
    i, j = ij[0], ij[1]
    # print('i, j = ', i, j)
    if (0 < i < max_n) and (0 < j < max_m):
        surroundings = [(ii, jj) for ii in [i - 1, i, i + 1]
                        for jj in [j - 1, j, j + 1]]
    if (i == 0) and (0 < j < max_m):
        surroundings = [(ii, jj) for ii in [i, i + 1]
                        for jj in [j - 1, j, j + 1]]
    if (i == max_n) and (0 < j < max_m):
        surroundings = [(ii, jj) for ii in [i - 1, i]
                        for jj in [j - 1, j, j + 1]]
    if (j == 0) and (0 < i < max_n):
        surroundings = [(ii, jj) for ii in [i - 1, i, i + 1]
                        for jj in [j, j + 1]]
    if (j == max_m) and (0 < i < max_n):
        surroundings = [(ii, jj) for ii in [i - 1, i, i + 1]
                        for jj in [j - 1, j]]
    if (i == 0) and (j == 0):
        surroundings = [(ii, jj) for ii in [i, i + 1]
                        for jj in [j, j + 1]]
    if (i == 0) and (j == max_m):
        surroundings = [(ii, jj) for ii in [i, i + 1]
                        for jj in [j - 1, j]]
    if (i == max_n) and (j == 0):
        surroundings = [(ii, jj) for ii in [i - 1, i]
                        for jj in [j, j + 1]]
    if (i == max_n) and (j == max_m):
        surroundings = [(ii, jj) for ii in [i - 1, i]
                        for jj in [j - 1, j]]
    return surroundings

class Grass(object):
    def __init__(self, map_shape, grass_options):
        self.N, self.M = map_shape[0], map_shape[1]
        self.options = grass_options
        self.grass_matrix = grass_options['initial_grass_level'] * np.ones((self.N, self.M)).astype(int)
        self.growth_rate = grass_options['growth_rate']

    def grow_grass(self):
        self.grass_matrix += self.growth_rate * np.ones((self.N, self.M)).astype(int)

class Rabbits(object):
    def __init__(self, map_shape, rabbit_options):
        self.N, self.M = map_shape[0], map_shape[1]
        self.options = rabbit_options
        self.generate_sample()

    def generate_sample(self):
        self.rabbit_matrix = np.zeros((self.N, self.M, 3)).astype(int)
        # Rabbit Matrix last dimension
        # 0: Alive (1) or Dead (0)
        # 1: Food status
        # 2: Age
        if self.options['seed'] != None:
            seed = self.options['seed']
        else:
            seed = np.random.randint(0, high=99999)
        initial_number = self.options['n_rabbits']
        index_list = [(i,j) for i in range(self.N) for j in range(self.M)]
        NM = np.arange(self.N*self.M)
        index_choice = RandomState(seed).choice(NM, initial_number)
        for ij in index_choice:
            i, j = index_list[ij]
            self.rabbit_matrix[i, j, 0] = 1
            # If you initialize all Rabbits with age 0 or equally hungry
            # they all die at once or start reproducing at the same time
            self.rabbit_matrix[i, j, 1] = np.random.randint(0, 4)
            self.rabbit_matrix[i, j, 2] = np.random.randint(0, 5)

    def next_move(self, surroundings, population_matrix):
        """
        :param surroundings: List of (i,j) of available surroundings
        :param population_matrix: Matrix containing both Rabbits & Wolves
        Check in the surroundings to find spots without
        Rabbits and without Wolves

        Returns: a list containing possible indices (i,j) for the next move
        """
        possible_moves = []
        for ij in surroundings:
            i, j = ij[0], ij[1]
            # Check if there's other rabbits
            if population_matrix[i,j,0] == 0:
                possible_moves.append(ij)
        if not possible_moves:
            # print('No moves')
            return []
        if possible_moves:
            # Randomly choose among possible moves
            choice = np.random.choice(np.arange(len(possible_moves)), 1)[0]
            next_i, next_j = possible_moves[choice][0], possible_moves[choice][1]
            return [next_i, next_j]

    def turn(self, grass_matrix, wolf_matrix):
        current_position = np.argwhere(self.rabbit_matrix[:,:,0] == 1)
        population_matrix = wolf_matrix + self.rabbit_matrix

        for ij in current_position:
            # print('\nRabbit #%d' %k)
            # Check whether there are rabbits nearby
            surroundings = compute_surroundings(ij, self.N, self.M)
            i, j = ij[0], ij[1]
            # print('Position: ', i, j)

            # Check if too old
            age = self.rabbit_matrix[i, j, -1].copy()
            if age > self.options['max_age']:
                # print('Dies')
                self.rabbit_matrix[i, j, :] = np.zeros(3)
                continue

            # Check if possible to reproduce
            age = self.rabbit_matrix[i, j, -1].copy()
            food_status = self.rabbit_matrix[i, j, 1].copy()
            if (age >= self.options['reproductive_age']) and (food_status >= self.options['min_food_to_repr']):
                # Possible to reproduce, check if available space
                nexts = self.next_move(surroundings=surroundings, population_matrix=population_matrix)
                if not nexts:
                    pass
                if nexts and (np.random.random() < self.options['reproductive_success']):
                    new_i, new_j = nexts[0], nexts[1]
                    # Give birth to a new rabbit
                    # print('A new rabbit is born')
                    self.rabbit_matrix[new_i, new_j, :] = np.array([1, 0, 0])

            self.rabbit_matrix[i, j, 1] -= self.options['hunger']

            # Check if Hungry
            food_status = self.rabbit_matrix[i,j,1].copy()
            if food_status < self.options['max_food_cap']:
                # Check if there's enough to eat
                if 0 < grass_matrix[i,j] < self.options['eating_rate']:
                    # Eats whatever is left
                    eats = grass_matrix[i,j]
                    # print('Eats: ', eats)
                    self.rabbit_matrix[i, j, 1] += eats
                    grass_matrix[i,j] -= eats

                elif grass_matrix[i,j] >= self.options['eating_rate']:
                    # Eats what it needs
                    eats = self.options['eating_rate']
                    # print('Eats: ', eats)
                    self.rabbit_matrix[i, j, 1] += eats
                    grass_matrix[i, j] -= eats

                elif grass_matrix[i,j] == 0:
                    # print('Nothing left to eat')
                    nexts = self.next_move(surroundings=surroundings, population_matrix=population_matrix)
                    if not nexts:
                        pass
                    if nexts:
                        next_i, next_j = nexts[0], nexts[1]
                        rabbit_status = self.rabbit_matrix[i, j, :].copy()
                        # print('Moves to: ', next_i, next_j)
                        # print(rabbit_status)
                        self.rabbit_matrix[next_i, next_j, :] = rabbit_status
                        # Clear the trail of the rabbit
                        self.rabbit_matrix[i, j, :] = np.zeros(3)

            # check if death by starvation
            food_status = self.rabbit_matrix[i,j,1].copy()
            if food_status < 0:
                self.rabbit_matrix[i, j, :] = np.zeros(3)
                continue

            # if it survives the whole iteration
            self.rabbit_matrix[i, j, -1] += 1

class Wolves(object):
    def __init__(self, map_shape, wolf_options):
        self.N, self.M = map_shape[0], map_shape[1]
        self.options = wolf_options
        self.generate_sample()

    def generate_sample(self):
        self.wolf_matrix = np.zeros((self.N, self.M, 3)).astype(int)
        if self.options['seed'] != None:
            seed = self.options['seed']
        else:
            seed = np.random.randint(0, high=99999)
        initial_number = self.options['n_wolves']
        index_list = [(i,j) for i in range(self.N) for j in range(self.M)]
        NM = np.arange(self.N*self.M)
        index_choice = RandomState(seed).choice(NM, initial_number)
        for ij in index_choice:
            i, j = index_list[ij]
            self.wolf_matrix[i, j, 0] = 1
            # If you initialize all Rabbits with age 0 or equally hungry
            # they all die at once or start reproducing at the same time
            self.wolf_matrix[i, j, 1] = np.random.randint(low=self.options['max_food_cap']//2,
                                                          high=3*self.options['max_food_cap']//4)
            self.wolf_matrix[i, j, 2] = np.random.randint(0, high=self.options['max_age']//2)

    def next_move(self, surroundings, population_matrix, status):
        """
        :param surroundings: List of (i,j) of available surroundings
        :param population_matrix: Matrix containing both Rabbits & Wolves
        :param status: whether to check for empty spaces (0) or full (0)
        Check in the surroundings to find spots with/without available spots

        Returns: a list containing possible indices (i,j) for the next move
        """
        possible_moves = []
        for ij in surroundings:
            i, j = ij[0], ij[1]
            # Check if there's other rabbits
            if population_matrix[i,j,0] == status:
                possible_moves.append(ij)
        if not possible_moves:
            # print('No moves')
            return []
        if possible_moves:
            # Randomly choose among possible moves
            choice = np.random.choice(np.arange(len(possible_moves)), 1)[0]
            next_i, next_j = possible_moves[choice][0], possible_moves[choice][1]
            return [next_i, next_j]

    def turn(self, rabbit_matrix, rabbit_food):
        current_position = np.argwhere(self.wolf_matrix[:,:,0] == 1)
        population_matrix = self.wolf_matrix + rabbit_matrix

        for ij in current_position:
            surroundings = compute_surroundings(ij, self.N, self.M)
            i, j = ij[0], ij[1]
            # Check if too old
            age = self.wolf_matrix[i, j, -1].copy()
            if age > self.options['max_age']:
                # print('Wolf dies of age')
                self.wolf_matrix[i, j, :] = np.zeros(3)
                continue

            # Check if possible to reproduce
            age = self.wolf_matrix[i, j, -1].copy()
            food_status = self.wolf_matrix[i, j, 1].copy()
            if (age >= self.options['reproductive_age']) and (food_status >= self.options['min_food_to_repr']):
                empty_space = self.next_move(surroundings=surroundings, population_matrix=population_matrix, status=0)
                if not empty_space:
                    pass
                if empty_space and (np.random.random() < self.options['reproductive_success']):
                    new_i, new_j = empty_space[0], empty_space[1]
                    # print('New Wolf is born')
                    self.wolf_matrix[new_i, new_j, :] = np.array([1, 0, 0])

            # Effect of hunger
            self.wolf_matrix[i, j, 1] -= self.options['hunger']

            # Check if Hungry
            food_status = self.wolf_matrix[i,j,1].copy()
            if food_status < self.options['max_food_cap']:
                # Check if there's a rabbit in the surroundings
                rabbit = self.next_move(surroundings=surroundings, population_matrix=rabbit_matrix, status=1)
                if not rabbit:
                    empty_space = self.next_move(surroundings=surroundings, population_matrix=population_matrix, status=0)
                    if not empty_space:
                        pass
                    if empty_space: # If not food but space around -> Move
                        next_i, next_j = empty_space[0], empty_space[1]
                        wolf_status = self.wolf_matrix[i, j, :].copy()
                        self.wolf_matrix[next_i, next_j, :] = wolf_status
                        self.wolf_matrix[i, j, :] = np.zeros(3)
                        # print('Wolf Moves')
                if rabbit: # Eat that rabbit!
                    rabbit_i, rabbit_j = rabbit[0], rabbit[1]
                    rabbit_matrix[rabbit_i, rabbit_j, :] = np.zeros(3)
                    # Update the nutritional value
                    self.wolf_matrix[i, j, 1] += rabbit_food
                    # print('Wolf kills a rabbit')

            # check if death by starvation
            food_status = self.wolf_matrix[i,j,1].copy()
            if food_status < 0:
                self.wolf_matrix[i, j, :] = np.zeros(3)
                # print('Wolf dies of Hunger')
                continue

            # if it survives the whole iteration
            self.wolf_matrix[i, j, -1] += 1


class Simulation(object):
    def __init__(self, N, M, grass_options, rabbit_options, wolf_options, max_iter):
        self.grass = Grass([N,M], grass_options)
        self.rabbits = Rabbits([N,M], rabbit_options)
        self.wolves = Wolves([N,M], wolf_options)
        self.max_iter = max_iter

        self.base_grass = np.sum(self.grass.grass_matrix.copy())
        self.max_spaces = N*M

    def simulate(self):
        self.grass_matrices, self.rabbit_matrices, self.wolf_matrices = [], [], []
        self.grass_level, self.n_rabits, self.n_wolves = [], [], []
        self.rabbit_ratio, self.wolf_ratio, self.animal_ratio = [], [], []

        for i in range(self.max_iter):
            print('\n-----------------------------------')
            print('Iter: ', i)

            """ Grass """
            current_grass = self.grass.grass_matrix.copy()
            self.grass_matrices.append(current_grass)
            total_grass = np.sum(current_grass)
            print('Grass: ', total_grass)
            self.grass_level.append(total_grass/ self.base_grass)

            """ Rabbits """
            current_rabbits = self.rabbits.rabbit_matrix.copy()
            self.rabbit_matrices.append(current_rabbits)
            total_rabbits = np.sum(current_rabbits[:,:,0])
            self.n_rabits.append(total_rabbits)
            rabbit_occupation = total_rabbits/self.max_spaces
            self.rabbit_ratio.append(rabbit_occupation)
            print('Rabbits: %d (%.2f)' %(total_rabbits, rabbit_occupation))

            """ Wolves """
            current_wolves = self.wolves.wolf_matrix.copy()
            self.wolf_matrices.append(current_wolves)
            total_wolves = np.sum(current_wolves[:,:,0])
            self.n_wolves.append(total_wolves)
            wolf_occupation = total_wolves/self.max_spaces
            self.wolf_ratio.append(wolf_occupation)
            print('Wolves: %d (%.2f)' %(total_wolves, wolf_occupation))

            print('Total: %d (%.2f)' % (total_wolves + total_rabbits, wolf_occupation + rabbit_occupation))

            self.rabbits.turn(self.grass.grass_matrix, self.wolves.wolf_matrix)
            self.wolves.turn(self.rabbits.rabbit_matrix, self.rabbits.options['nutritional_value'])
            self.grass.grow_grass()


    def plot_populations(self):
        plt.figure()
        iters = np.arange(self.max_iter)
        plt.scatter(iters, self.n_rabits, color='blue', s=3)
        plt.xlabel('Iteration')
        plt.ylim([0, max(self.n_rabits)])
        plt.title('Rabbit population')

        plt.figure()
        plt.scatter(iters, self.n_wolves, color='red', s=3)
        plt.xlabel('Iteration')
        plt.ylim([0, max(self.n_wolves)])
        plt.title('Wolf population')

        plt.figure()
        plt.scatter(iters, self.grass_level, color='green', s=3)
        plt.ylim([0, max(self.grass_level)])
        plt.xlabel('Iteration')
        plt.title('Ratio of grass/grass0')

        plt.figure()
        plt.plot(iters, self.rabbit_ratio, color='blue', label='Rabbits')
        plt.plot(iters, self.wolf_ratio, color='red', label='Wolves')
        plt.ylim([0., 1.])
        plt.legend()
        plt.xlabel('Iteration')
        plt.title('Occupation ratios')


    def plot_results(self, iter, current_grass, current_rabbits):
        plt.figure()
        plt.imshow(current_rabbits[:, :, 0])
        plt.title('Rabbits at iteration %d' %iter)

        plt.figure()
        plt.imshow(current_grass, vmin=0, cmap='Greens')
        plt.colorbar()
        plt.title('Grass at iteration %d' %iter)

    def show_grass(self, i):
        return self.grass_matrices[i]

    def plot_animation(self):
        fig = plt.figure()
        grass_ims = []
        for i in range(self.max_iter):
            im = plt.imshow(self.show_grass(i), vmin=0, animated=True, cmap='Greens')
            grass_ims.append(im)
        ani = animation.ArtistAnimation(fig, grass_ims, interval=50,
                                        blit=True, repeat_delay=1000)
        # ani.save('Grass.mp4')



if __name__ == "__main__":
    # Parameters
    N, M = 20, 20       # Map dimensions
    max_iter = 500      # Number of iterations
    grass_options = {'initial_grass_level':10, 'growth_rate':1}
    rabbit_options = {'n_rabbits':100, 'max_food_cap':45, 'eating_rate':5,
                      'hunger':1,
                      'reproductive_age':10, 'reproductive_success':0.75,
                      'min_food_to_repr':5, 'max_age':30,
                      'nutritional_value':10, 'seed':None}

    wolf_options = {'n_wolves':5, 'max_food_cap':200, 'hunger':3,
                      'reproductive_age':15, 'reproductive_success':0.5,
                      'min_food_to_repr':25, 'max_age':30, 'seed':None}

    sim = Simulation(N, M, grass_options, rabbit_options, wolf_options, max_iter)
    sim.simulate()
    sim.plot_populations()


    # plt.figure()
    # plt.imshow(sim.rabbits.rabbit_matrix[:,:,0].copy())
    # plt.title('Initial Rabbits')
    #
    # plt.figure()
    # plt.imshow(sim.grass.grass_matrix.copy(), vmin=0, cmap='Greens')
    # plt.colorbar()
    # plt.title('Initial Grass')

    # sim.rabbits.turn(sim.grass.grass_matrix)

    # plt.figure()
    # plt.imshow(sim.rabbits.rabbit_matrix[:,:,0].copy())
    # plt.title('New Rabbits')
    #
    # plt.figure()
    # plt.imshow(sim.grass.grass_matrix.copy(), vmin=0, cmap='Greens')
    # plt.colorbar()
    # plt.title('New Grass')


    plt.show()