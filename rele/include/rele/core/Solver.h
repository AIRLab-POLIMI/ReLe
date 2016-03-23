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

#ifndef INCLUDE_RELE_CORE_SOLVER_H_
#define INCLUDE_RELE_CORE_SOLVER_H_

#include "rele/core/Transition.h"
#include "rele/core/Core.h"
#include "rele/core/PolicyEvalAgent.h"
#include "rele/policy/Policy.h"

namespace ReLe
{

/*!
 * A solver is the abstract interface for exact or approximate solvers for a specific family of environments.
 * Differently from agents, it does not implement a logger interface, and doesn't need to interface with a core.
 * Often a solver is declared as friend of the class of MDPs that can solve, to access it's internal state.
 */
template<class ActionC, class StateC>
class Solver
{
public:
    /*!
     * Default Constructor
     */
    Solver()
    {
        testEpisodeLength = 100;
        testEpisodes = 1;
    }

    /*!
     * This method run the computation of the optimal policy for the MDP.
     * Must be implemented.
     */
    virtual void solve() = 0;

    /*!
     * This method runs the policy computed by Solver::solve() over the MDP.
     * Must be implemented. Normally the implementation is simply wrapper for
     * Solver::test(Environment<ActionC, StateC>& env, Policy<ActionC, StateC>& pi)
     * \return the set of trajectories sampled with the optimal policy from the MDP
     */
    virtual Dataset<ActionC, StateC> test() = 0;

    /*!
     * Getter
     * \return the set the optimal policy from the MDP
     */
    virtual Policy<ActionC, StateC>& getPolicy() = 0;

    /*!
     * Set the episodes number and the episode maximum length
     * \see Solver::test()
     * \param testEpisodes the number of test episodes to run
     * \param testEpisodeLength the maximum length for each test episode
     */
    inline void setTestParams(unsigned int testEpisodes,
                              unsigned int testEpisodeLength)
    {
        this->testEpisodeLength = testEpisodeLength;
        this->testEpisodes = testEpisodes;
    }

    /*!
     * Destructor
     */
    virtual ~Solver()
    {
    }

protected:
    /*!
     * This method implements the low level test using the core.
     */
    virtual Dataset<ActionC, StateC> test(Environment<ActionC, StateC>& env,
                                          Policy<ActionC, StateC>& pi)
    {
        PolicyEvalAgent<ActionC, StateC> agent(pi);
        Core<ActionC, StateC> core(env, agent);

        CollectorStrategy<ActionC, StateC> strategy;
        core.getSettings().loggerStrategy = &strategy;
        core.getSettings().episodeLength = testEpisodeLength;
        core.getSettings().testEpisodeN = testEpisodes;

        core.runTestEpisodes();

        return strategy.data;
    }

protected:
    unsigned int testEpisodeLength;
    unsigned int testEpisodes;

};

}

#endif /* INCLUDE_RELE_CORE_SOLVER_H_ */
