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

#include "Solver.h"
#include "ParametricRewardMDP.h"

namespace ReLe
{

template<class ActionC, class StateC>
class IRLSolver: public Solver<ActionC, StateC>
{
public:
    IRLSolver(Environment<ActionC, StateC>& mdp, AbstractBasisMatrix& features,
              ParametricRegressor& rewardRegressor) :
        prMdp(mdp, rewardRegressor), features(features),
        rewardRegressor(rewardRegressor)
    {

    }

    inline void setWeights(arma::vec& weights)
    {
        rewardRegressor.setParameters(weights);
    }

    inline unsigned int getBasisSize()
    {
        return features.rows();
    }

    inline double getGamma()
    {
        return prMdp.getSettings().gamma;
    }

    arma::mat computeFeaturesExpectations()
    {
        Dataset<ActionC, StateC> && data = this->test();
        return data.computefeatureExpectation(features, prMdp.getSettings().gamma);
    }

protected:
    ParametricRewardMDP<ActionC, StateC> prMdp;
    AbstractBasisMatrix& features;
    ParametricRegressor& rewardRegressor;

};

template<class ActionC, class StateC>
class IRLAgentSolver: public IRLSolver<ActionC, StateC>
{

public:
    IRLAgentSolver(Agent<ActionC, StateC>& agent,
                   Environment<ActionC, StateC>& mdp,
                   Policy<ActionC, StateC>& policy, AbstractBasisMatrix& features,
                   ParametricRegressor& rewardRegressor) :
        IRLSolver<ActionC, StateC>(mdp, features, rewardRegressor), agent(agent), policy(policy)
    {
        episodes = 1;
        episodeLength = 10000;
    }

    virtual void solve()
    {
        EmptyStrategy<ActionC, StateC> strategy;
        Core<ActionC, StateC> core(this->prMdp, agent);
        core.getSettings().episodeLenght = episodeLength;
        core.getSettings().episodeN = episodes;
        core.getSettings().loggerStrategy = &strategy;

        core.runEpisodes();
    }

    virtual Dataset<ActionC, StateC> test()
    {
        return Solver<ActionC, StateC>::test(this->prMdp, policy);
    }

    virtual Policy<ActionC, StateC>& getPolicy()
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
