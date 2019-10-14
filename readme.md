SMB3_RNN-Neat_Python

A small script used with Retro-Gym (OpenAI) that will run along side of the NEAT algorithm which evaluates mario's performance and determines if Mario can complete the level. The inputs to the Nueral Network is just the pixel data from the emulator. This means that the network will learn what to do based on what the network recognizes.

- main.py is for a single instance of the emulator that will be rendered and can be viewed while it runs. A ton of statistics will be in the console pertaining to each generation that passes as mario learns.

- main_parallel.py is for multiple iterations of the emululator being ran at once. This speeds up the process immensly. A ton of statistics will be in the console pertaining to each generation that passes as mario learns.

- fitness_test.lua is the fitness formula, just written in lua. This was used for testing the fitness formula before deployment to make sure that it makes sense. You can load this into an emulator and test it out see the fitness formula in real time.

- neat-config.txt is the configuration file that the network uses to build a population. You can tweak this to try and pin-point better settings that can improve the performance of the network.

To get this running, you will need
- Retro-Gym
- Numpy
- OpenCV
- NEAT-Python
- Pickle
- OS