#include "rele/core/FiniteMDP.h"
#include "rele/algorithms/td/SARSA.h"
#include "rele/algorithms/td/Q-Learning.h"
#include "rele/core/Core.h"

#include "rele/policy/q_policy/e_Greedy.h"
#include "rele/policy/q_policy/Boltzmann.h"

#include "rele/generators/SimpleChainGenerator.h"

#include "rele/approximators/basis/IdentityBasis.h"

#include <iostream>

using namespace std;
using namespace ReLe;

int main(int argc, char *argv[])
{
    //Create the MDP
    SimpleChainGenerator generator;
    generator.generate(5, 2);
    FiniteMDP mdp = generator.getMDP(0.9);

    //Create the agent
    e_Greedy policy;
    ConstantLearningRate alpha(0.2);
    Q_Learning agent(policy, alpha);

    //Setup the experiment
    Core<FiniteAction, FiniteState> core(mdp, agent);
    core.getSettings().episodeLength = 10000;
    bool logTransition = false;
    bool logAgent = true;
    core.getSettings().loggerStrategy =
                new PrintStrategy<FiniteAction, FiniteState>(logTransition,
                                                             logAgent);

    //Run the learning
    cout << "starting episode" << endl;
    core.runEpisode();
    delete core.getSettings().loggerStrategy;

    return 0;
}
