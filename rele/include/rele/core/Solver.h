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

template<class ActionC, class StateC>
class Solver
{
public:
    Solver()
    {
        testEpisodeLength = 100;
        testEpisodes = 1;
    }

    virtual void solve() = 0;
    virtual Dataset<ActionC, StateC> test() = 0;
    virtual Policy<ActionC, StateC>& getPolicy() = 0;

    inline void setTestParams(unsigned int testEpisodes,
                              unsigned int testEpisodeLength)
    {
        this->testEpisodeLength = testEpisodeLength;
        this->testEpisodes = testEpisodes;
    }

    virtual ~Solver()
    {
    }

protected:
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
