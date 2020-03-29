# Deep Reinforcement Learning for Bao
This was a hobby project by @Dave-Frank-419 and me to explore the topic of deep reinforcement learning a bit.
## The game
We decided to tackle the mancala game Bao la kujifunza, a simple variant of Bao which is played on a board with 4x8 pits.
![bao_board](https://user-images.githubusercontent.com/9535190/77852183-861d4e00-71dd-11ea-9ced-74e93a2b4414.jpg)
In this variant there are 2 stones in each of the pits at the start of the game. The game rules can be found in the
[Wikipedia article](https://en.wikipedia.org/wiki/Bao_(game)), though we modified them a bit. For example our implementation
only allows clockwise sowing. It is apparent that our game variant doesn't allow nearly as much strategic depth as the more
complex variants of this game, but we just wanted a simple game to test the concepts and some algorithms of this field.
## Our approach
First of all we implemented the game as a gym environment and extracted some statistics from a couple of games simulated
with random moves. Then we proceeded to implement a basic reinforcement algorithm using a neural network for the game state
evaluation function. We didn't spend a lot of time on choosing the right architecture for the network and optimizing it though.
### The agent
We used a simple minimax search to make the agent take the values of future states in consideration and made it more efficient
by applying alpha-beta.pruning. Furthermore we decided to use a probability based cap instead of limiting the search depth
by simply capping the depth at a fixed value. Therefore our agent expands its search tree until the probability of the state
currently looked at being part of the optimal game tree falls below a certain threshold. After applying the softmax function
to the estimated state values from the neural network we interpreted the resulting values as the probabilities of these
actions being optimal in the current game state.
### The network
After trying a simple convnet we just settled on a simple fully connected dense network taking the one-hot encoded board
as its input.
### The training process
Not spending much time on the training process we simply let the agent train by playing against itself for about half an
hour without adjusting the learning rate or doing some other fancy stuff.
## Evaluation
The trained agent was able to beat an agent playing random moves in 1000 out of 1000 games with a fairly high probability
threshold of 50% resulting in a very shallow search. Though it has to be mentioned that after examining the agent's evaluation
of the game states we noticed there were still many temporal inconsistencies. Also we think that our game variant benefits
greatly from just calculating the different possible states and maybe the application of a neural network wasn't really
necessary. We are still satisfied with our results as the concept worked out quite well despite the code being far from
optimal.
