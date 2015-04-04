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

#include "Core.h"
#include "PolicyEvalAgent.h"

#include "FiniteMDP.h"
#include "SimpleChainGenerator.h"

#include "td/SARSA.h"
#include "q_policy/e_Greedy.h"

#include "ParametricRewardMDP.h"

#include "algorithms/MWAL.h"

using namespace ReLe;

class SimpleChainOptimalPolicy: public NonParametricPolicy<FiniteAction, FiniteState>
{
public:
    virtual unsigned int operator()(size_t state)
    {
        if(state > 1)
            return 1;
        else
            return 0;
    }

    virtual double operator()(size_t state, unsigned int action)
    {
        return 0;
    }

    inline virtual std::string getPolicyName()
    {
        return "Simple chain optimal";
    }

    virtual std::string getPolicyHyperparameters()
    {
        return "";
    }

    virtual std::string printPolicy()
    {
        return "";
    }

    virtual ~SimpleChainOptimalPolicy() {}
};

class SimpleChainBasis : public BasisFunction
{

public:
    SimpleChainBasis(size_t state) : state(state)
    {

    }

    virtual double operator()(const arma::vec& input)
    {
        size_t currentState = input[0];

        if(currentState == state)
            return 1;
        else
            return 0;
    }

    virtual void writeOnStream(std::ostream& out)
    {

    }

    virtual void readFromStream(std::istream& in)
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

    FiniteMDP mdp = generator.getMPD(0.9);

    SimpleChainOptimalPolicy expertPolicy;
    PolicyEvalAgent<FiniteAction, FiniteState> expert(expertPolicy);

    /* Generate expert dataset */
    Core<FiniteAction, FiniteState> expertCore(mdp, expert);
    CollectorStrategy<FiniteAction, FiniteState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLenght = 50;
    expertCore.getSettings().testEpisodeN = 1000;
    expertCore.runTestEpisodes();


    /* Learn weight with MWAL */

    //Create features vector
    DenseBasisVector rewardBasis;
    for(int i = 0; i < 5; i++)
    {
        SimpleChainBasis* bf = new SimpleChainBasis(i);
        rewardBasis.push_back(bf);
    }

    //Compute expert feature expectations
    arma::vec muE = collection.data.computefeatureExpectation(rewardBasis, mdp.getSettings().gamma);

    // Create a parametric MDP
    LinearApproximator rewardRegressor(1, rewardBasis);
    ParametricRewardMDP<FiniteAction, FiniteState> prMdp(mdp, rewardRegressor);

    //Create an agent to solve the mdp direct problem
    e_Greedy policy;
    SARSA agent(policy);

    //Setup the solver
    Core<FiniteAction, FiniteState> core(prMdp, agent);
    core.getSettings().episodeLenght = 1000;
    core.getSettings().episodeN = 1000;
    core.getSettings().testEpisodeN = 1000;

    //Run MWAL
    unsigned int T = 20;
    double gamma = prMdp.getSettings().gamma;

    MWAL<FiniteAction, FiniteState> irlAlg(rewardBasis, rewardRegressor, core, T, gamma, muE);
    irlAlg.run();
    arma::vec w = irlAlg.getWeights();

    std::cout << policy.printPolicy() << std::endl;


    return 0;
}
