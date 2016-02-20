/*
 * rele,
 *
 *
 * Copyright (C) 2016 Davide Tateo & Matteo Pirotta
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

#ifndef INCLUDE_RELE_CORE_BATCHCORE_H_
#define INCLUDE_RELE_CORE_BATCHCORE_H_

#include "rele/core/BatchAgent.h"

namespace ReLe
{

template<class ActionC, class StateC>
class BatchCore
{
public:
    struct CoreSettings
    {
        CoreSettings()
        {
            loggerStrategy = nullptr;
            maxIterations = 1;
            epsilon = 0.01;
        }

        LoggerStrategy<ActionC, StateC>* loggerStrategy;
        unsigned int maxIterations;
        double epsilon;
    };

public:
    BatchCore(Dataset<ActionC, StateC>& data, BatchAgent<ActionC, StateC>& agent) :
        data(data),
        agent(agent)
    {
    }

    CoreSettings& getSettings()
    {
        return settings;
    }

    void runEpisode()
    {
        //core setup
        //Logger<ActionC, StateC> logger;

        //logger.setStrategy(settings.loggerStrategy);

        //Start episode
        agent.init(data);

        for(unsigned int i = 0;
                i < settings.maxIterations
                && !agent.isTerminalConditionReached(); i++)
        {
            agent.step();
            //logger.log(agent.getAgentOutputData(), i);
        }

        //logger.log(agent.getAgentOutputDataEnd(), settings.episodeLength);
        //logger.printStatistics();
    }

    /*void runSteps()
    {
        for(unsigned int i = 0; i < settings.maxIterations; i++)
        {
            runEpisode();
        }
    }*/

protected:
    Dataset<ActionC, StateC>& data;
    BatchAgent<ActionC, StateC>& agent;
    CoreSettings settings;
};

template<class ActionC, class StateC>
BatchCore<ActionC, StateC> buildCore(Dataset<ActionC, StateC>& data,
                                     BatchAgent<ActionC, StateC>& agent)
{
    return BatchCore<ActionC, StateC>(data, agent);
}

}

#endif /* INCLUDE_RELE_CORE_BATCHCORE_H_ */
