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

#include "rele/core/Core.h"
#include "rele/core/PolicyEvalAgent.h"
#include "rele/core/BatchAgent.h"
#include "rele/utils/FileManager.h"

#include <iostream>

namespace ReLe
{

template<class ActionC, class StateC>
class BatchOnlyCore
{
public:
    struct BatchOnlyCoreSettings
    {
        BatchOnlyCoreSettings()
        {
            loggerStrategy = nullptr;
            envName = "env";
            algName = "alg";
            maxBatchIterations = 1;
        }

        LoggerStrategy<ActionC, StateC>* loggerStrategy;
        std::string envName;
        std::string algName;
        unsigned int maxBatchIterations;
    };

public:
    BatchOnlyCore(Dataset<ActionC, StateC> data,
                  BatchAgent<ActionC, StateC>& batchAgent) :
        data(data),
        batchAgent(batchAgent)
    {
    }

    BatchOnlyCoreSettings& getSettings()
    {
        return settings;
    }

    void run()
    {
        //core setup
        Logger<ActionC, StateC> logger;

        logger.setStrategy(settings.loggerStrategy);

        //Start episode
        batchAgent.init(data);

        for(unsigned int i = 0;
                i < settings.maxBatchIterations
                && !batchAgent.hasConverged(); i++)
        {
            batchAgent.step();
            logger.log(batchAgent.getAgentOutputData(), i);
        }

        //logger.log(agent.getAgentOutputDataEnd(), settings.maxIterations);
        //logger.printStatistics();
    }

protected:
    Dataset<ActionC, StateC> data;
    BatchAgent<ActionC, StateC>& batchAgent;
    BatchOnlyCoreSettings settings;
};

template<class ActionC, class StateC>
class BatchCore
{

public:
    struct BatchCoreSettings
    {
        BatchCoreSettings()
        {
            loggerStrategy = nullptr;
            nTransitions = 100;
            episodeLength = 100;
            nEpisodes = 100;
            envName = "env";
            algName = "alg";
            datafileName = "dataset.csv";
            maxBatchIterations = 1;
        }

        LoggerStrategy<ActionC, StateC>* loggerStrategy;
        std::string envName;
        std::string algName;
        std::string datafileName;
        unsigned int nTransitions;
        unsigned int episodeLength;
        unsigned int nEpisodes;
        unsigned int maxBatchIterations;
    };

public:
    BatchCore(Environment<ActionC, StateC>& mdp,
    		  BatchAgent<ActionC, StateC>& batchAgent) :
        mdp(mdp),
        batchAgent(batchAgent)
    {
    }

    BatchCoreSettings& getSettings()
    {
        return settings;
    }

    void run(Policy<ActionC, StateC>& policy)
    {
    	PolicyEvalAgent<ActionC, StateC> agent(policy);

        auto&& core = buildCore(mdp, agent);

        CollectorStrategy<ActionC, StateC> collection;
        core.getSettings().loggerStrategy = &collection;

        core.getSettings().episodeLength = settings.nTransitions;
        core.getSettings().testEpisodeN = settings.nEpisodes;

        core.runTestEpisodes();

        Dataset<ActionC, StateC> data = collection.data;

        FileManager fm(settings.envName, settings.algName);
        fm.createDir();
        fm.cleanDir();
        std::ofstream out(fm.addPath(settings.datafileName), std::ios_base::out);
        out << std::setprecision(OS_PRECISION);
        if(out.is_open())
            data.writeToStream(out);
        out.close();

        std::cout << std::endl << "# Ended data collection and save" << std::endl << std::endl;

        auto&& batchCore = buildBatchOnlyCore(data, batchAgent);

        batchCore.getSettings().loggerStrategy = settings.loggerStrategy;
        batchCore.getSettings().envName = settings.envName;
        batchCore.getSettings().algName = settings.algName;
        batchCore.getSettings().maxBatchIterations = settings.maxBatchIterations;

        batchCore.run();
    }


protected:
    Environment<ActionC, StateC>& mdp;
    BatchAgent<ActionC, StateC>& batchAgent;
    BatchCoreSettings settings;
};

template<class ActionC, class StateC>
BatchCore<ActionC, StateC> buildBatchCore(
    Environment<ActionC, StateC>& mdp,
    BatchAgent<ActionC, StateC>& batchAgent)
{
    return BatchCore<ActionC, StateC>(mdp, batchAgent);
}

template<class ActionC, class StateC>
BatchOnlyCore<ActionC, StateC> buildBatchOnlyCore(
    Dataset<ActionC, StateC> data,
    BatchAgent<ActionC, StateC>& batchAgent)
{
    return BatchOnlyCore<ActionC, StateC>(data, batchAgent);
}

}

#endif /* INCLUDE_RELE_CORE_BATCHCORE_H_ */
