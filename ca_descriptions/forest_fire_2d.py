# Name: Forest Fire Simulation
# Dimensions: 2

# --- Set up executable path, do not edit ---
import sys
import inspect
import random
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


def transition_func(grid, neighbourstates, neighbourcounts, decaygrid, initial_terrain, topology_grid):

    def decision(probability):
        return random.random() < probability

    def probability_p_burn(terrain):
        p_h, p_veg, p_den, p_w, p_s = (0, 0, 0, 0, 0)
        if terrain == 0:
            p_h, p_veg, p_den, p_w, p_s = (0.05, 0.5, 0.5, 0.5, 0.5)
        elif terrain == 2:
            p_h, p_veg, p_den, p_w, p_s = (0.05, 0.05, 0.05, 0.5, 0.5)
        elif terrain == 3:
            p_h, p_veg, p_den, p_w, p_s = (1, 1, 1, 1, 1)
        return p_h*(1+p_veg)*(1+p_den)*p_w*p_s

    def probability_p_w(wind_direction, cell_coordindates):
        pass

    def fire_direction(cell_coordinates, grid):
        pass

    # unpack the state arrays
    NW, N, NE, W, E, SW, S, SE = neighbourstates

    print(NW.shape)

    # select wind direction in degrees
    wind_direction = 150

    # cells that can catch fire: chaparral, forest, scrubland
    burnable_cells = ((grid == 0) | (grid == 2) | (grid == 3))

    # cells that were burning in last time step
    burning_cells = (grid == 5)

    # cells that have 1 or more neighbours in state 5 (on fire)
    cells_with_burning_neighbours = (neighbourcounts[5] >= 1)

    # take one off their decay value, calculates for how many time steps a cell has been burning
    decaygrid[burning_cells] -= 1

    # assigns the probability of each cell catching fire and then probabilistically decides if it will
    for row in range(0, 199):
        for col in range(0, 199):
            if col:
                if row == 0 or col == 0:
                    cells_with_burning_neighbours[col][row] = False
                elif cells_with_burning_neighbours[col][row]:
                    cells_with_burning_neighbours[col][row] = decision(probability_p_burn(grid[col][row]))

    # prevents fire from spreading on the other side of the map
    cells_with_burning_neighbours[0] = False
    cells_with_burning_neighbours[199] = False

    # cells that are burnable and have been probabilistically assigned as catching fire will catch fire
    to_fire_state = burnable_cells & cells_with_burning_neighbours

    # for how many generations can certain terrain burn
    chaparral_burning_gen = -1000
    dense_forest_burning_gen = -50
    scrubland_burning_gen = -10

    decayed_to_burned_land = (((decaygrid == chaparral_burning_gen) & (initial_terrain == 0))
                              | ((decaygrid == dense_forest_burning_gen) & (initial_terrain == 2))
                              | ((decaygrid == scrubland_burning_gen) & (initial_terrain == 3)))

    grid[decayed_to_burned_land] = 4
    grid[to_fire_state] = 5

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

    config.grid_dims = (200, 200)
    config.num_generations = 500

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

    # initialise the decay grid
    decaygrid = np.zeros(config.grid_dims)
    decaygrid.fill(2)

    # Initialise topology Grid
    max_height = 10



    noise = np.random.normal(0, max_height, np.zeros(config.grid_dims))
    topology_grid = abs(noise).astype(int)


    # Create grid object
    grid = Grid2D(config, (transition_func, decaygrid, initial_terrain, topology_grid))

    # Run the CA, save grid state every generation to timeline
    timeline = grid.run()

    # save updated config to file
    config.save()
    # save timeline to file
    utils.save(timeline, config.timeline_path)


if __name__ == "__main__":
    main()
