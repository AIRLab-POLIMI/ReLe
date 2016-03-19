==================
Learn a simple MDP
==================

In this tutorial we will use ReLe to run the Q-Learning algorithm on a simple MDP.

We will use as environment a simple chain. (TODO describe environment)

First of all we will create a Finite MDP using a generator:

TODO CODE

Now we have to create a Q-Learning agent. The Q-Learning agent needs a policy specification, and the learning rate.
For this simple environment we can use an \f$\epsilon\f$-greedy policy and a costant learning rate

TODO CODE

Finally we create a core to run our agent on the mdp.
In this simple example we can just run a single episode.
We will use a PrintStrategy (TODO LINK) to print the results on the console.

TODO CODE


After running the code you shuold see on the console two section:

- statistics, in wich is described the initial state and the state frequencies
- Agent data at episode end, in wich is printed agent data:
-- Parameters: discount factor, learning rate, \f$\epsilon\f$ of the epsilon greedy
-- Action Value function
-- Policy


The output shoudl be similar to this:

(TODO format)

	starting episode


	--- statistics ---

	- Initial State
	x(t0) = [3]
	- State Statistics
	0: 0.0165
	1: 0.2389
	2: 0.4696
	3: 0.2574
	4: 0.0176

	--- Agent data at episode end ---
	Using e-Greedy policy

	- Parameters
	gamma: 0.9
	alpha: 0.200000
	eps: 0.15
	- Action-value function
	Q(0, 0) = 3.89557
	Q(0, 1) = 3.13971
	Q(1, 0) = 4.30185
	Q(1, 1) = 3.62914
	Q(2, 0) = 3.58693
	Q(2, 1) = 3.7851
	Q(3, 0) = 3.56217
	Q(3, 1) = 4.3094
	Q(4, 0) = 2.89102
	Q(4, 1) = 3.84717
	- Policy
	policy(0) = 0
	policy(1) = 0
	policy(2) = 1
	policy(3) = 1
	policy(4) = 1


Please, note that this mdp has two optimal policies, as in the goal state (state n 2), the two action has the same expected return.
