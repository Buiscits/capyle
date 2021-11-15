# Name: Forest Fire Simulation
# Dimensions: 2

# --- Set up executable path, do not edit ---
import sys
import inspect
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


def transition_func(grid, neighbourstates, neighbourcounts):

    # NOTE THESE ARE STILL THE DEFAULT VALUES FROM GAME OF LIFE

    # dead = state == 0, live = state == 1
    # unpack state counts for state 0 and state 1
    dead_neighbours, live_neighbours = neighbourcounts
    # create boolean arrays for the birth & survival rules
    # if 3 live neighbours and is dead -> cell born
    birth = (live_neighbours == 3) & (grid == 0)
    # if 2 or 3 live neighbours and is alive -> survives
    survive = ((live_neighbours == 2) | (live_neighbours == 3)) & (grid == 1)
    # Set all cells to 0 (dead)
    grid[:, :] = 0
    # Set cells to 1 where either cell is born or survives
    grid[birth | survive] = 1
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

    config.state_colors = [chaparral_color, lake_color, dense_forest_color, scrubland_color, town_color, fire_color]

    config.grid_dims = (200, 200)
    # config.num_generations = 150

    def draw_terrain():
        rows, cols = config.grid_dims

        # fill everything with chaparral
        arr = [[0] * cols] * rows
        arr = np.array(arr)

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

        print(arr)
        return arr

    config.set_initial_grid(draw_terrain())

    # ----------------------------------------------------------------------

    if len(args) == 2:
        config.save()
        sys.exit()

    return config


def main():
    # Open the config object
    config = setup(sys.argv[1:])

    # Create grid object
    grid = Grid2D(config, transition_func)

    # Run the CA, save grid state every generation to timeline
    timeline = grid.run()

    # save updated config to file
    config.save()
    # save timeline to file
    utils.save(timeline, config.timeline_path)


if __name__ == "__main__":
    main()
