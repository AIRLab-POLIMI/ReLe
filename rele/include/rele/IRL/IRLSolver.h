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

#ifndef INCLUDE_RELE_IRL_IRLSOLVER_H_
#define INCLUDE_RELE_IRL_IRLSOLVER_H_

#include "rele/core/Solver.h"
#include "rele/IRL/ParametricRewardMDP.h"

namespace ReLe
{

template<class ActionC, class StateC, class FeaturesInputC = arma::vec>
class IRLSolver: public Solver<ActionC, StateC>
{
public:
    IRLSolver(Features_<FeaturesInputC>& phi) : phi(phi)
    {

    }

    virtual void setWeights(arma::vec& weights) = 0;
    virtual double getGamma() = 0;

    inline unsigned int getBasisSize()
    {
        return phi.rows();
    }

    virtual arma::mat computeFeaturesExpectations()
    {
        Dataset<ActionC, StateC> && data = this->test();
        return data.computefeatureExpectation(phi, this->getGamma());
    }


    virtual ~IRLSolver()
    {

    }

protected:
    Features_<FeaturesInputC>& phi;

};

template<class ActionC, class StateC, class FeaturesInputC = arma::vec>
class IRLSolverBase: public IRLSolver<ActionC, StateC>
{
public:
    IRLSolverBase(Environment<ActionC, StateC>& mdp, Features_<FeaturesInputC>& phi,
                  ParametricRegressor_<FeaturesInputC>& rewardRegressor) :
        IRLSolver<ActionC, StateC, FeaturesInputC>(phi),
        prMdp(mdp, rewardRegressor),
        rewardRegressor(rewardRegressor)
    {

    }

    inline void setWeights(arma::vec& weights) override
    {
        rewardRegressor.setParameters(weights);
    }

    inline double getGamma() override
    {
        return prMdp.getSettings().gamma;
    }

protected:
    ParametricRewardMDP<ActionC, StateC> prMdp;
    ParametricRegressor_<FeaturesInputC>& rewardRegressor;

};

template<class ActionC, class StateC, class FeaturesInputC = arma::vec>
class IRLAgentSolver: public IRLSolverBase<ActionC, StateC, FeaturesInputC>
{

public:
    IRLAgentSolver(Agent<ActionC, StateC>& agent,
                   Environment<ActionC, StateC>& mdp,
                   Policy<ActionC, StateC>& policy, Features_<FeaturesInputC>& features,
                   ParametricRegressor_<FeaturesInputC>& rewardRegressor) :
        IRLSolverBase<ActionC, StateC, FeaturesInputC>(mdp, features, rewardRegressor),
        agent(agent), policy(policy)
    {
        episodes = 1;
        episodeLength = 10000;
    }

    virtual void solve() override
    {
        Core<ActionC, StateC> core(this->prMdp, agent);
        core.getSettings().episodeLength = episodeLength;
        core.getSettings().episodeN = episodes;
        core.getSettings().loggerStrategy = nullptr;

        core.runEpisodes();
    }

    virtual Dataset<ActionC, StateC> test() override
    {
        return Solver<ActionC, StateC>::test(this->prMdp, policy);
    }

    virtual Policy<ActionC, StateC>& getPolicy() override
    {
        return policy;
    }

    inline void setLearningParams(unsigned int episodes,
                                  unsigned int episodeLength)
    {
        this->episodeLength = episodeLength;
        this->episodes = episodes;
    }

    virtual ~IRLAgentSolver()
    {

    }

protected:
    Agent<ActionC, StateC>& agent;
    Policy<ActionC, StateC>& policy;

    unsigned int episodeLength;
    unsigned int episodes;

};

}

#endif /* INCLUDE_RELE_IRL_IRLSOLVER_H_ */
