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
#include "rele/core/logger/BatchAgentLogger.h"
#include "rele/core/logger/BatchDatasetLogger.h"

#include <iostream>

namespace ReLe
{

/*!
 * This class can be used to run a batch agent over a dataset.
 * This class handles batch agent termination flag and logging.
 * The maximum number f iterations and the logging strategy can be specified in the settings.
 */
template<class ActionC, class StateC>
class BatchOnlyCore
{
public:
    /*!
     * This struct contains the core settings
     */
    struct BatchOnlyCoreSettings
    {
        BatchOnlyCoreSettings()
        {
            logger = nullptr;
            maxBatchIterations = 1;
        }

        //! The logger for agent data
        BatchAgentLogger<ActionC, StateC>* logger;
        //! The maximum number of iteration of the algorithm over the dataset
        unsigned int maxBatchIterations;
    };

public:
    /*!
     * Constructor.
     * \param data the dataset used for batch learning
     * \param batchAgent a batch learning agent
     */
    BatchOnlyCore(Dataset<ActionC, StateC> data,
                  BatchAgent<ActionC, StateC>& batchAgent) :
        data(data),
        batchAgent(batchAgent)
    {
    }

    /*!
     * Getter.
     * \return the settings of the core.
     */
    BatchOnlyCoreSettings& getSettings()
    {
        return settings;
    }

    /*!
     * Run the batch iterations over the dataset specified in the settings.
     * \param envSettings settings of the environment
     */
    void run(EnvironmentSettings envSettings)
    {
        //Start episode
        batchAgent.init(data, envSettings);

        for(unsigned int i = 0;
                i < settings.maxBatchIterations
                && !batchAgent.hasConverged(); i++)
        {
            batchAgent.step();

            if(settings.logger)
                if(!batchAgent.hasConverged() && i < settings.maxBatchIterations)
                    settings.logger->log(batchAgent.getAgentOutputData(), i);
                else
                    settings.logger->log(batchAgent.getAgentOutputDataEnd(), i);
        }
    }

protected:
    Dataset<ActionC, StateC> data;
    BatchAgent<ActionC, StateC>& batchAgent;
    BatchOnlyCoreSettings settings;
};


/*!
 * This class is an extension of BatchOnlyCore, that takes care not only to run the batch algorithm,
 * but also to generate the dataset and test the learned policy over the environment.
 */
template<class ActionC, class StateC>
class BatchCore
{

public:
    /*!
     * This struct contains the core settings
     */
    struct BatchCoreSettings
    {
        BatchCoreSettings()
        {
            datasetLogger = nullptr;
            agentLogger = nullptr;
            episodeLength = 100;
            nEpisodes = 100;
            maxBatchIterations = 1;
        }

        //! The logger for the dataset
        BatchDatasetLogger<ActionC, StateC>* datasetLogger;
        //! The logger for agent data
        BatchAgentLogger<ActionC, StateC>* agentLogger;
        //! The episode lenght
        unsigned int episodeLength;
        //! The number of episodes to run
        unsigned int nEpisodes;
        //! The maximum number of algorithm iterations.
        unsigned int maxBatchIterations;
    };

public:
    /*!
     * Constructor.
     * \param environment the environment used by this experiment
     * \param batchAgent the batch learning agent
     */
    BatchCore(Environment<ActionC, StateC>& environment,
              BatchAgent<ActionC, StateC>& batchAgent) :
        environment(environment),
        batchAgent(batchAgent)
    {
    }

    /*!
     * Getter.
     * \return the core settings.
     */
    BatchCoreSettings& getSettings()
    {
        return settings;
    }

    /*!
     * This method is used to generate a dataset from the environment and
     * run the batch learning algorithm on it.
     */
    void run(Policy<ActionC, StateC>& policy)
    {
        Dataset<ActionC, StateC>&& data = test(&policy);

        if(settings.datasetLogger)
            settings.datasetLogger->log(data);

        auto&& batchCore = buildBatchOnlyCore(data, batchAgent);

        batchCore.getSettings().logger = settings.agentLogger;
        batchCore.getSettings().maxBatchIterations = settings.maxBatchIterations;

        batchCore.run(environment.getSettings());
    }

    /*!
     * This method generates a dataset using the agent learned policy
     * \return the generated dataset
     */
    Dataset<ActionC, StateC> runTest()
    {
        return test(batchAgent.getPolicy());
    }

protected:
    Environment<ActionC, StateC>& environment;
    BatchAgent<ActionC, StateC>& batchAgent;
    BatchCoreSettings settings;

protected:
    Dataset<ActionC, StateC> test(Policy<ActionC, StateC>* policy)
    {
        PolicyEvalAgent<ActionC, StateC> agent(*policy);

        auto&& core = buildCore(environment, agent);

        CollectorStrategy<ActionC, StateC> collection;
        core.getSettings().loggerStrategy = &collection;

        core.getSettings().episodeLength = settings.episodeLength;
        core.getSettings().testEpisodeN = settings.nEpisodes;

        core.runTestEpisodes();

        return collection.data;
    }
};


/*!
 * This function can be used to get a BatchOnlyCore instance from an agent and an environment,
 * reducing boilerplate code:
 * Example:
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
 * auto&& batchOnlyCore = buildBatchOnlyCore(environment, agent);
 * core.run();
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 */
template<class ActionC, class StateC>
BatchOnlyCore<ActionC, StateC> buildBatchOnlyCore(
    Dataset<ActionC, StateC> data,
    BatchAgent<ActionC, StateC>& batchAgent)
{
    return BatchOnlyCore<ActionC, StateC>(data, batchAgent);
}

/*!
 * This function can be used to get a BatchCore instance from an agent and an environment,
 * reducing boilerplate code:
 * Example:
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
 * auto&& batchCore = buildBatchCore(environment, agent);
 * core.run();
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
template<class ActionC, class StateC>
BatchCore<ActionC, StateC> buildBatchCore(
    Environment<ActionC, StateC>& mdp,
    BatchAgent<ActionC, StateC>& batchAgent)
{
    return BatchCore<ActionC, StateC>(mdp, batchAgent);
}


}

#endif /* INCLUDE_RELE_CORE_BATCHCORE_H_ */
