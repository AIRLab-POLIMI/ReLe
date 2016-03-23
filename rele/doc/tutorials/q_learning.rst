==================
Learn a simple MDP
==================

In this tutorial we will use ReLe to run the Q-Learning algorithm on a simple MDP.

We will use as environment a simple chain below:

.. image:: images/SimpleChain.svg
            
The action success probability is p=0.8, if an action fails the agent stays in the same state.                      

First of all we will create a Finite MDP using a generator:

.. literalinclude:: code/q_learning.cpp
   :language: c++
   :linenos:
   :lines: 20-23

Now we have to create a Q-Learning agent. The Q-Learning agent needs a policy specification, and the learning rate.
For this simple environment we can use an :math:`\epsilon`-greedy policy and a costant learning rate

.. literalinclude:: code/q_learning.cpp
   :language: c++
   :linenos:
   :lines: 25-28

Finally we create a core to run our agent on the mdp.
In this simple example we can just run a single episode.
We will use a :cpp:class:`ReLe::PrintStrategy` to print the results on the console.

.. literalinclude:: code/q_learning.cpp
   :language: c++
   :linenos:
   :lines: 30-41


After running the code you should see on the console two section:

- Statistics, in which is described the initial state and the state frequencies
- Agent data at episode end, in which is printed agent data

  - Parameters: discount factor, learning rate, :math:`\epsilon` of the :math:`\epsilon` greedy policy
  - Action Value function
  - Policy


The output shoudl be similar to this::

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


Please, note that this mdp has two optimal policies, as in the goal state (state n 2), the two action has the same expected return; and Q-Learning can find two different optimal policies.

The complete code is the following:

.. literalinclude:: code/q_learning.cpp
   :language: c++
   :linenos:

