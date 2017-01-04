# DQN-DDQN-on-Space-Invaders
Implementation of Double Deep Q Networks and Dueling Q Networks using Keras on Space Invaders using OpenAI Gym. Code can be easily generalized to other Atari games.

## Prerequistes
You can install all the prerequistes for code use using 

```text
pip install -r requirements.txt
```

## Instructions on Use
Details about the code are covered in the blog [here](https://yilundu.github.io/2016/12/24/Deep-Q-Learning-on-Space-Invaders.html)

To run the code use
```python
  python main.py
```
with arguments where arguments are given by

```text
usage: main.py [-h] -n NETWORK -m MODE [-l LOAD] [-s SAVE] [-x] [-v]

Train and test different networks on Space Invaders

optional arguments:
  -h, --help            show this help message and exit
  -n NETWORK, --network NETWORK
                        Please specify the network you wish to use, either DQN
                        or DDQN
  -m MODE, --mode MODE  Please specify the mode you wish to run, either train
                        or test
  -l LOAD, --load LOAD  Please specify the file you wish to load weights
                        from(for example saved.h5)
  -s SAVE, --save SAVE  Specify folder to render simulation of network in
  -x, --statistics      Specify to calculate statistics of network(such as
                        average score on game)
  -v, --view            Display the network playing a game of space-invaders.
                        Is overriden by the -s command
  ```
  
  For example, to test the pre-trained Double Deep Q Network architecture and view the network playing space invaders use
  
  ```text
    python main.py -n DDQN -m test -l saved.h5 -v
  ```
  
 or to train the Dueling Q Network architecture and then save the resulting video of the network playing in the test/ directory 
 use 
 
   ```text
    python main.py -n DQN -m train -s test
  ```
 

 *Note that as the model is trained, every 10000 images, the program saves the network weights in either saved.h5 of duel_saved.h5 for DDQN and DQN respectively*
