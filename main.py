import retro
import neat
import numpy as np
import cv2
import pickle
import os

# Create an environment in gym to run the emulator
# We are using FCUEX through Gym-Retro to play SMB3
# World 1 - Level 1 specifically
env = retro.make('SuperMarioBros3-Nes', '1Player.World1.Level1.state')


# The run command is called from main to create the population from
# the config file and then evaluate that population
def run(path, prev_model=False):
    # Check if we have a previous model to evaluate
    if prev_model:
        # Create a list of checkpoints
        checkpoints = []
        # Loop through the files in the root directory
        for file in os.listdir(path):
            # If the file is a neat-checkpoint
            if file.find('neat-checkpoint') is not -1:
                # add it to our list of checkpoints
                checkpoints.append(file)
        if len(checkpoints) > 0:
            # Order the checkpoints from highest number (most recent) to lowest number
            checkpoints.sort(reverse=True)
            # Print a message for the checkpoint that we will be using
            print("Found a checkpoint file:", checkpoints[0])
            # Load the checkpoint into a population (called 'pop')
            pop = neat.Checkpointer.restore_checkpoint(checkpoints[0])
        else:
            # Otherwise ..
            print("No checkpoint file found. Loading a new population.")
            # The config file is loaded in with the specific headers we are using
            # All this can be found in the 'neat-config.txt' file
            config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                        path + "/neat-config.txt")
            # create a new population with the config file
            pop = neat.Population(config)
    else:
        # Otherwise ..
        # The config file is loaded in with the specific headers we are using
        # All this can be found in the 'neat-config.txt' file
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                    path + "/neat-config.txt")
        # create a new population with the config file
        pop = neat.Population(config)

    # Create a reporter to show statistics and add it to the population
    pop.add_reporter(neat.StdOutReporter(True))
    # Create our statistics reporter
    stats = neat.StatisticsReporter()
    # Add our stats reporter to the population
    pop.add_reporter(stats)
    # Add a checkpointer, which will save checkpoints every 10 generations
    pop.add_reporter(neat.Checkpointer(generation_interval=10))
    # The winner (the highest fitness after the fitness goal has been achieved)
    # is saved into a variable
    winner = pop.run(eval_genomes)
    # and then written to a model file for safe keeping
    pickle.dump(winner, open("model.pk", "wb"))


# Evaluating the genomes function
def eval_genomes(genomes, config):
    # For each genome, we will do the following:
    for genome_id, genome in genomes:
        # Initialize the environment for each genome
        ob = env.reset()
        # Calculate the inputs for the network to accept:
        # Start with the shape of the screen (in pixels)
        # X and Y are the rows and columns of the screen
        # Z is the color depth of the pixel at (x, y)
        screen_x, screen_y, screen_z = env.observation_space.shape
        # Store the screen size by one fifth of its original size
        screen_x = int(screen_x / 5)
        # Do this for the Y as well.
        screen_y = int(screen_y / 5)

        # Create a new network for the genome to attempt and find outputs
        net = neat.nn.RecurrentNetwork.create(genome, config)
        # Initialize a few things for later
        # The max fitness achieved by this genome
        max_fitness = 0
        # Current attempt's fitness
        fitness = 0
        # Frame count variable for later for when multiple frames have
        # passed and no progress is being made
        frame_count = 0

        # xpos is the current position (in pixels) for mario
        xpos = 24
        # ypos is the current position (in pixels) for mario
        ypos = -128
        # max_pos is the furthest this genome has gotten (in pixels)
        max_pos = [xpos, ypos]
        # prev_pos is the position of mario from the last frame
        prev_pos = [xpos, ypos]
        # Conveniently max_pos and prev_pos start out the same
        # this will be used to verify that mario is still on the move
        # every frame

        # game_score is the initial value in RAM for the score displayed
        # we can work this into fitness as well and reward the genome for
        # killing enemies and getting coin blocks
        game_score = 0
        # game_time is the initial value in RAM for the time displayed
        # they are three different bytes, on for the 100 slot, one for
        # the 10 slot, and one for the last digit.
        game_time = 300

        # The done variable is used to stop the genome if our goal is achieved
        # or we have stopped progressing.
        done = False

        # While we haven't achieved our goal, run frame by frame.
        while not done:
            # Render out the game so we can visualize what's happening
            env.render()

            # Get the frame of the game and set it up to input into the network
            # Resize the screen to the size we defined earlier
            ob = cv2.resize(ob, (screen_x, screen_y))
            # Convert the color to black and white. Color isn't super important here
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            # Reshape the array of pixel values to match our expected screen size
            ob = np.reshape(ob, (screen_x, screen_y))
            # Flatten the array to make it one-dimensional (NEAT required the input to be 1D)
            ob = ob.flatten()
            # BOOM! Activate these inputs against the networks configuration
            nn_Output = net.activate(ob)

            # Step the emulator one frame and get some information from that frame:
            # ob is the observable screen
            # rew is the current reward as of this frame
            # done is the flag for if our goal was met
            # ram_val contains information from RAM (specified in another file)
            ob, rew, done, ram_val = env.step(nn_Output)

            # ram_val contains several values directly from RAM inside the emulator
            # currently, we are working with the following:
            # ------------------------------------------------------------------------
            # Name          Address         Use
            # ------------------------------------------------------------------------
            # goal_card     0x7D9C          Check if we have received a goal card
            # lives         0x0736          Number of lives remaining (We die?)
            # score         0x0715          Score we have received from playing
            # screen_num    0x0075          Which screen the player is currently on
            # time_0xx      0x05EE          First digit (hundreds) of the timer
            # time_x0x      0x05EF          Second digit (tens) of the timer
            # time_xx0      0x05F0          Last digit (ones) of the timer
            # x_pos         0x0090          X coordinate of the player (on the screen)
            # y_pos         0x00A2          Y coordinate of the player (on the screen)

            # This next set of 'if / else' statements are checked each frame
            # to determine the state of the game based on the env.step() function
            # and we reward fitness based on a few parameters

            # First thing we check is if mario moved
            # cur_pos will be initialized each frame with the current position of mario.
            # We will keep track of our X Coordinate by adding the x_pos and the screen number
            # together. screen_num is essentially a counter for every time mario goes beyond
            # 256 pixels to the right.
            cur_pos = [ram_val['x_pos'] + (ram_val['screen_num'] * 256), ram_val['y_pos']]
            # Verify the cur_pos is not the same as the prev_pos
            if cur_pos is not prev_pos:
                # if our current X value is higher than the previous X value
                if cur_pos[0] > prev_pos[0]:
                    # Reward two fitness
                    fitness += 2
                # if our current X value is higher than our maximum position
                if cur_pos[0] > max_pos[0]:
                    # record our new max as out current position
                    max_pos[0] = cur_pos[0]
                # if our current Y value is higher than our maximum position
                if cur_pos[1] > max_pos[1]:
                    # record our new max as out current position
                    max_pos[1] = cur_pos[1]
                # lastly set our prev_pos to our cur_pos for reference next frame
                prev_pos = cur_pos

            # We check to see if our lives are less than the initial 4
            # basically just a flag if we died
            if ram_val['lives'] is not 4:
                # mark this genome as complete as he can't continue
                done = True

            # We check if the game score is higher than we previously knew
            if ram_val['score'] > game_score:
                # if so, we add the score difference between last frame and this frame
                fitness += ram_val['score'] - game_score
                # then record the current score as the actual score in RAM
                game_score = ram_val['score']

            # We check the game time on every frame and as long as it is not 0, we keep track
            # of it with out game_time variable which will work into our fitness score when we
            # lose a life or complete the level.
            if ((ram_val['time_0xx'] * 100) + (ram_val['time_x0x'] * 10) + ram_val['time_xx0']) is not 0:
                # set the game_time equal to what the current values are in RAM
                game_time = ((ram_val['time_0xx'] * 100) + (ram_val['time_x0x'] * 10) + ram_val['time_xx0'])

            # We check if we got a goal card at the end of the level
            # basically if we completed the objective
            if ram_val['goal_card'] is not 0:
                # reward the threshold for out fitness (defined in the config file)
                # which completed the objective entirely
                fitness += 125000
                # mark this genome as done as we've completed the goal
                done = True

            # We check if our current fitness is higher than last frame
            if fitness > max_fitness:
                # keep frame_count at 0 as we still can get further
                frame_count = 0
                # record the new max_fitness
                max_fitness = fitness
            # otherwise ..
            else:
                # we increment frame_count since we aren't progressing any
                frame_count += 1

            # We finally check if done is true OR if frame_count is 500
            # if done was marked true, either by the emulator or one
            # of our conditions .. or if 500 frames have passed with
            # no progression
            if frame_count == 500 or done:
                # Mark this genome as done (used for the frame_count part of the condition)
                done = True
                # Add to our fitness the amount of time we had left at the end of
                # the attempt and divide it by three to weight it a little better
                fitness += game_time

            # Lastly, record this genomes fitness for the network
            # with the fitness we achieved
            genome.fitness = fitness


# Main function
if __name__ == "__main__":
    # Grab the local directory for this script
    local_dir = os.path.dirname(__file__)
    # make a path to the config file based on the local path
    config_path = os.path.join(local_dir)
    # run the network with the config file we found
    run(config_path, True)
