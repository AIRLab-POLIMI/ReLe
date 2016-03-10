/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo & Matteo Pirotta
 * Versione 1.0
 *
 * This file is part of rele.
 *
 * rele is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * rele is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with rele.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "rele/core/Core.h"
#include "rele/core/PolicyEvalAgent.h"

#include "rele/core/FiniteMDP.h"
#include "rele/generators/SimpleChainGenerator.h"
#include "rele/solvers/dynamic_programming/BasicDynamicProgramming.h"

#include "rele/algorithms/td/SARSA.h"
#include "rele/policy/q_policy/e_Greedy.h"

#include "rele/approximators/features/DenseFeatures.h"

#include "rele/IRL/ParametricRewardMDP.h"

#include "rele/IRL/algorithms/MWAL.h"

using namespace ReLe;

class SimpleChainBasis : public BasisFunction
{

public:
    SimpleChainBasis(size_t state) : state(state)
    {

    }

    virtual double operator()(const arma::vec& input) override
    {
        size_t currentState = input[2];

        if(currentState == state)
            return 1;
        else
            return 0;
    }

    virtual void writeOnStream(std::ostream& out) override
    {

    }

    virtual void readFromStream(std::istream& in) override
    {

    }

private:
    size_t state;

};

int main(int argc, char *argv[])
{
    /* Create simple chain and optimal policy */
    SimpleChainGenerator generator;
    generator.generate(5, 2);

    FiniteMDP mdp = generator.getMDP(0.9);

    PolicyIteration expertSolver(mdp);
    expertSolver.solve();
    PolicyEvalAgent<FiniteAction, FiniteState> expert(expertSolver.getPolicy());

    /* Generate expert dataset */
    expertSolver.setTestParams(1000, 50);
    Dataset<FiniteAction, FiniteState>&& data = expertSolver.test();


    /* Learn weight with MWAL */

    //Create features vector
    BasisFunctions rewardBasis;
    for(int i = 0; i < 5; i++)
    {
        SimpleChainBasis* bf = new SimpleChainBasis(i);
        rewardBasis.push_back(bf);
    }

    DenseFeatures rewardPhi(rewardBasis);
    LinearApproximator rewardRegressor(rewardPhi);

    //Compute expert feature expectations
    arma::vec muE = data.computefeatureExpectation(rewardPhi, mdp.getSettings().gamma);

    //Create an agent to solve the mdp direct problem
    e_Greedy policy;
    ConstantLearningRate alpha(0.2);
    SARSA agent(policy, alpha);

    //Setup the solver
    IRLAgentSolver<FiniteAction, FiniteState> solver(agent, mdp, policy, rewardPhi, rewardRegressor);
    solver.setLearningParams(1000, 1000);
    solver.setTestParams(100, 10000);

    //Run MWAL
    unsigned int T = 20;

    MWAL<FiniteAction, FiniteState> irlAlg(T,  muE, solver);
    irlAlg.run();
    arma::vec w = irlAlg.getWeights();

    std::cout << policy.printPolicy() << std::endl;


    return 0;
}
