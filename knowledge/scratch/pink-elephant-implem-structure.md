# Pink Elephant implementation structure

1. convert board state -> tensor. build it in such a way that it is easy to flip state between player (current/opponent/empty)

2. code up network that takes in that tensor as input and gives output
- the output will be two things a polciy head and a value head
- policy head tells you the distribution on next moves and value head tells you the current chance you have to win rn 

the whole point is that:
- the value head helps you not search all the way down the thing it is way faster


when we are performing MCTS:
- we pick the thing that argmaxxes puct
- then we go down and then we keep going down until we hit a leaf node
- then we start the next simulation
- then we keep going and going until we explore
- then we take the distrubtion over next moves and our goal is to make the current distribution into that distribution because it was search enhanced and we want to see how good it is

so with this, we train both the vlaue head and the polciy head at the same time and they involve the same 



code the network up. batch the examples and after palying one game we train on those examples/results
value_pred -> matches the end result
policy -> matches the policy for current board state

===
implem order:
1. define the board state -> tensor 
2. code up the network
3. download games expert games from lichess + train on those games (they will live on the computer), just about downloading them and setting htem up in a way that is easy to parse
4. train network
5. then self play loop setup
6. use go for callbacks (eventully to speed up)
7. leaf parallelism
8. game parallelism

===

I want to be using modal for a lot of the mcts stuff and just storage of things. specifcaly like modal volume + containers for running the mcts stuff. also like when I do inference I want to store model weights on modal volume + also store checkpoints + like idk i want to run inference on modal like cheap gpu type thing

^ overarching plan about how I want to implement this.


