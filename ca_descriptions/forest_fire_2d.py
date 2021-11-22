# Name: Forest Fire Simulation
# Dimensions: 2

# --- Set up executable path, do not edit ---
import sys
import inspect
import random
import scipy.stats

this_file_loc = (inspect.stack()[0][1])
main_dir_loc = this_file_loc[:this_file_loc.index('ca_descriptions')]
sys.path.append(main_dir_loc)
sys.path.append(main_dir_loc + 'capyle')
sys.path.append(main_dir_loc + 'capyle/ca')
sys.path.append(main_dir_loc + 'capyle/guicomponents')
# ---   

from capyle.ca import Grid2D, Neighbourhood, CAConfig, randomise2d
import capyle.utils as utils
import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter


def transition_func(grid, neighbourstates, neighbourcounts, decaygrid, initial_terrain, topology_grid, count):
    count[0] = count[0] + 1

    terrain_fire_rates = {0: 0.05, 1: 0, 2: 0.05, 3: 1, 4: 0, 5: 0}

    def lightning_strike():
        x, y = (random.randint(0, 199), random.randint(0, 199))
        if probability_p_burn(y, x, lighting=True):
            grid[y][x] = 5


    def find_random_burning_cell(array):
        (burning_x, burning_y) = random.choice(array)
        return burning_x, burning_y

    def spotting(burning_cell, wind_direction, wind_speed):
        x, y = burning_cell
        distance_x, distance_y = x*wind_direction[0], y*wind_direction[1]

        if distance_x < 0:
            distance_x *= -1
        elif distance_x > 0:
            distance_x = 200 - distance_x

        if distance_y < 0:
            distance_y *= -1
        elif distance_y > 0:
            distance_y = 200 - distance_y

        relative_dist_x = random.randint(0, int(distance_x * (wind_speed / 25)))
        relative_dist_y = random.randint(0, int(distance_y * (wind_speed / 25)))

        x = x + relative_dist_x*wind_direction[0]
        y = y + relative_dist_y*wind_direction[1]

        if probability_p_burn(y, x, spotted=True):
            grid[y][x] = 5

    def decision(probability):
        return random.random() < probability
    
    def probability_p_burn(x, y, spotted=False, lighting=False):
        terrain = grid[x][y]
        p_h, p_veg, p_den, p_w, p_s = (0, 0, 0, 0.0, 0)

        if spotted or lighting:
            if terrain == 0:
                p_h, p_veg, p_den = (0.5, 0.5, 0.5)
            elif terrain == 2:
                p_h, p_veg, p_den = (0.05, 0.05, 0.05)
            elif terrain == 3:
                p_h, p_veg, p_den = (1, 1, 1)
            decide = decision(p_h * (1 + p_veg) * (1 + p_den))
            return decide

        wind_probabilities = probability_p_w([1, 0], 5.0, fire_direction(x, y))
        p_s = calculate_slope_fire_spread_probability(x, y)

        decide = False

        for p_w in wind_probabilities:
            if terrain == 0:
                p_h, p_veg, p_den, p_s = (0.5, 0.5, 0.5, p_s)
            elif terrain == 2:
                p_h, p_veg, p_den, p_s = (0.05, 0.05, 0.05, p_s)
            elif terrain == 3:
                p_h, p_veg, p_den, p_s = (1, 1, 1, 1)

            decide = decision(p_h * (1 + p_veg) * (1 + p_den) * p_w * p_s)
            if decide:
                break

        return decide

    def probability_p_w(wind_direction, wind_speed, relative_fire_coordinates):
        x_vect, y_vect = wind_direction
        wind_direction = (y_vect, x_vect*(-1))
        wind_probabilities = []

        for fire_direction in relative_fire_coordinates:
            angle = angle_between_two_vectors(fire_direction, wind_direction)/2
            pdf = scipy.stats.norm(0, 1/wind_speed).pdf(angle)
            wind_probabilities.append(pdf)

        return wind_probabilities

    def calculate_slope_fire_spread_probability(row, column):

        # Calculate final fire rate probability due to elevation
        neighbour_terrain_fire_spread_probabilities = calculate_terrain_spread_probabilities(row, column)
        #print("MAX: ", max(neighbour_terrain_fire_spread_probabilities))
        # See if slope small enough to make block catch fire
        on_fire_threshold = 0.1
        cells_that_should_spread_fire = np.array(neighbour_terrain_fire_spread_probabilities) < on_fire_threshold

        #print("LEN:", (cells_that_should_spread_fire == True).size / 8)
        return (cells_that_should_spread_fire == True).size / 8

    def calculate_terrain_spread_probabilities(row, column):
        
        def calculate_slope(current_cell, neighbour_cell, is_diagonal):
            if is_diagonal:
                return (current_cell, math.degrees(math.atan((current_cell - neighbour_cell) / math.sqrt(2))))
            else:
                return (current_cell, math.degrees(math.atan((current_cell - neighbour_cell) / 1)))

        terrain = grid[row][column]
        
         # Non Diagonal
        up_slope = calculate_slope(terrain, topology_grid[row][col + 1], False)
        down_slope = calculate_slope(terrain, topology_grid[row][col - 1], False)
        right_slope = calculate_slope(terrain, topology_grid[row + 1][col], False)
        left_slope = calculate_slope(terrain, topology_grid[row - 1][col], False)

        # Diagonal
        top_left_slope = calculate_slope(terrain, topology_grid[row - 1][col - 1], True)
        bottom_left_slope = calculate_slope(terrain, topology_grid[row + 1][col - 1], True)
        top_right_slope = calculate_slope(terrain, topology_grid[row - 1][col + 1], True)
        bottom_right_slope = calculate_slope(terrain, topology_grid[row + 1][col + 1], True)

        slopes = [up_slope, down_slope, left_slope, right_slope, top_left_slope, top_right_slope, bottom_left_slope, bottom_right_slope]

        fire_probabilities = list(map(lambda x: terrain_fire_rates[x[0]] * math.exp(1 * x[1]), slopes))

        return fire_probabilities
    
    def angle_between_two_vectors(v1, v2):
        def unit_vector(vector):
            return np.divide(vector, np.linalg.norm(vector))

        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def fire_direction(cell_x, cell_y):
        relative_fire_coordinates = []
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                if cell_x + x < 200 and cell_y + y < 200:
                    if grid[cell_x + x][cell_y + y] == 5 and grid[cell_x][cell_y] != 5:
                        relative_fire_coordinates.append((x, y))

        return relative_fire_coordinates

    np.seterr(invalid='ignore')  # ignores warnings lol
    #grid[y axis][x axis]

    # cells that can catch fire: chaparral, forest, scrubland and have a neighbour thats on fire
    burnable_cells = (((grid == 0) | (grid == 2) | (grid == 3)) & (neighbourcounts[5] >= 1))

    # cells that were burning in last time step
    burning_cells = (grid == 5)

    # take one off their decay value, calculates for how many time steps a cell has been burning
    decaygrid[burning_cells] -= 1
    # assigns the probability of each cell catching fire and then probabilistically decides if it will
    burning_array = []
    for row in range(0, 199):
        for col in range(0, 199):
            if grid[col][row] == 5:
                burning_array.append((row, col))
            if row == 0 or col == 0 or burnable_cells[col][row] is False:
                burnable_cells[col][row] = False
            elif burnable_cells[col][row]:
                burnable_cells[col][row] = probability_p_burn(col, row)

    print(count - 1)
    # prevents fire from spreading on the other side of the map
    burnable_cells[0] = False
    burnable_cells[199] = False

    # for how many generations can certain terrain burn
    chaparral_burning_gen = -30
    dense_forest_burning_gen = -50
    scrubland_burning_gen = -10

    decayed_to_burned_land = (((decaygrid == chaparral_burning_gen) & (initial_terrain == 0))
                              | ((decaygrid == dense_forest_burning_gen) & (initial_terrain == 2))
                              | ((decaygrid == scrubland_burning_gen) & (initial_terrain == 3)))

    grid[decayed_to_burned_land] = 4
    grid[burnable_cells] = 5

    if decision(0.01):
        spotting(find_random_burning_cell(burning_array), [1, 0], 20)
    if decision(0.01):
        lightning_strike()

    return grid


def setup(args):
    config_path = args[0]
    config = utils.load(config_path)
    # ---THE CA MUST BE RELOADED IN THE GUI IF ANY OF THE BELOW ARE CHANGED---
    config.title = "Forest Fire Simulator"
    config.dimensions = 2
    config.states = (0, 1, 2, 3, 4, 5)
    # ------------------------------------------------------------------------

    # ---- Override the defaults below (these may be changed at anytime) ----
    # color codes for each terrain property

    chaparral_color = (229, 208, 176)
    lake_color = (75, 182, 239)
    dense_forest_color = (70, 85, 82)
    scrubland_color = (151, 151, 96)
    town_color = (0, 0, 0)
    fire_color = (247, 55, 24)

    config.state_colors = np.array([chaparral_color, lake_color, dense_forest_color, scrubland_color, town_color, fire_color])/255

    grid_height = 200
    grid_width = 200

    config.grid_dims = (grid_width, grid_height)
    config.num_generations = 100

    def draw_terrain():
        rows, cols = config.grid_dims

        # fill everything with chaparral
        arr = [[0] * cols] * rows
        arr = np.array(arr)

        #add fire
        x1, x2 = int(0.08*cols), int(0.085*cols)
        y1, y2 = rows-int(0.65*rows), rows-int(0.645*rows)
        arr[y1:y2, x1:x2] = 5


        # add lakes
        x1, x2 = int(0.15*cols), int(0.2*cols)
        y1, y2 = rows-int(0.7*rows), rows-int(0.6*rows)
        arr[y1:y2, x1:x2] = 1

        x1, x2 = int(0.6*cols), int(0.9*cols)
        y1, y2 = rows-int(0.35*rows), rows-int(0.3*rows)
        arr[y1:y2, x1:x2] = 1

        # add dense forest
        x1, x2 = int(0.2*cols), int(0.3*cols)
        y1, y2 = rows-int(0.6*rows), rows-int(0.2*rows)
        arr[y1:y2, x1:x2] = 2

        x1, x2 = int(0.3*cols), int(0.4*cols)
        y1, y2 = rows-int(0.9*rows), rows-int(0.4*rows)
        arr[y1:y2, x1:x2] = 2

        x1, x2 = int(0.3*cols), int(1*cols)
        y1, y2 = rows-int(0.5*rows), rows-int(0.4*rows)
        arr[y1:y2, x1:x2] = 2

        # add scrubland
        x1, x2 = int(0.25*cols), int(0.3*cols)
        y1, y2 = rows-int(0.9*rows), rows-int(0.1*rows)
        arr[y1:y2, x1:x2] = 3

        # add a town
        x1, x2 = int(0.5*cols), int(0.55*cols)
        y1, y2 = rows-int(0.25*rows), rows-int(0.2*rows)
        arr[y1:y2, x1:x2] = 4

        return arr

    initial_terrain = draw_terrain()
    config.set_initial_grid(draw_terrain())

    # ----------------------------------------------------------------------

    if len(args) == 2:
        config.save()
        sys.exit()

    return config, initial_terrain


def main():
    # Open the config object
    config, initial_terrain = setup(sys.argv[1:])
    count = np.array([0])

    # initialise the decay grid
    decaygrid = np.zeros(config.grid_dims)
    decaygrid.fill(2)

    # Initialise topology Grid
    max_height = 100
 
    noise = np.random.normal(0, max_height, size=config.grid_dims)
    topology_grid = abs(noise).astype(int)
    smoothed_topology = gaussian_filter(topology_grid, sigma=10)

    # Graph and Save the topology grid as png
    plt.imshow(smoothed_topology, cmap='hot', interpolation='nearest')
    fig1 = plt.gcf()
    plt.colorbar()
    plt.show()
    plt.draw()
    fig1.savefig('test.png', dpi=100)

    # Create grid object
    grid = Grid2D(config, (transition_func, decaygrid, initial_terrain, smoothed_topology, count))

    # Run the CA, save grid state every generation to timeline
    timeline = grid.run()

    # save updated config to file
    config.save()
    # save timeline to file
    utils.save(timeline, config.timeline_path)


if __name__ == "__main__":
    main()
