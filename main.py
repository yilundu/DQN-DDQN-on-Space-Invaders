import argparse
from space_invaders import SpaceInvader

# Hyperparameters
NUM_FRAME = 1000000

parser = argparse.ArgumentParser(description="Train and test different networks on Space Invaders")

# Parse arguments
parser.add_argument("-n", "--network", type=str, action='store', help="Please specify the network you wish to use, either DQN or DDQN", required=True)
parser.add_argument("-m", "--mode", type=str, action='store', help="Please specify the mode you wish to run, either train or test", required=True)
parser.add_argument("-l", "--load", type=str, action='store', help="Please specify the file you wish to load weights from(for example saved.h5)", required=False)
parser.add_argument("-s", "--save", type=str, action='store', help="Specify folder to render simulation of network in", required=False)
parser.add_argument("-x", "--statistics", action='store_true', help="Specify to calculate statistics of network(such as average score on game)", required=False)
parser.add_argument("-v", "--view", action='store_true', help="Display the network playing a game of space-invaders. Is overriden by the -s command", required=False)

args = parser.parse_args()
print(args)

game_instance = SpaceInvader(args.network)

if args.load:
    game_instance.load_network(args.load)

if args.mode == "train":
    game_instance.train(NUM_FRAME)

if args.statistics:
    stat = game_instance.calculate_mean()
    print("Game Statistics")
    print(stat)

if args.save:
    game_instance.simulate(path=args.save, save=True)
elif args.view:
    game_instance.simulate()

