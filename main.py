import argparse

parser = argparse.ArgumentParser(description="Train and test different networks on Space Invaders")

# Parse arguments
parser.add_argument("-n", "--network", type=str, action='store', help="Please specify the network you wish to use, either DQN or DDQN", required=True)
parser.add_argument("-m", "--mode", type=str, action='store', help="Please specify the mode you wish to run, either train or test", required=True)
parser.add_argument("-l", "--load", type=str, action='store', help="Please specify the file you wish to load weights from(for example saved.h5)", required=False)
parser.add_argument("-s", "--save", type=str, action='store', help="Specify folder to render simulation of network in", required=False)
parser.add_argument("-x", "--statistics", type=str, action='store', help="Specify to calculate statistics of network(such as average score on game)", required=False)
parser.add_argument("-v", "--view", type=str, action='store', help="Display the network playing a game of space-invaders. Is overriden by the -s command", required=False)

args = parser.parse_args()
print args
# print(args.accumulate(args.integers))
