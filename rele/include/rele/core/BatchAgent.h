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

#ifndef INCLUDE_RELE_ALGORITHMS_BATCH_BATCHAGENT_H_
#define INCLUDE_RELE_ALGORITHMS_BATCH_BATCHAGENT_H_

#include "rele/approximators/data/BatchData.h"
#include "rele/core/Transition.h"
#include "rele/approximators/Regressors.h"

namespace ReLe
{

/*!
 * The BatchAgent is the basic interface of all batch agents.
 * All batch algorithms should extend this abstract class.
 * The BatchAgent interface provides all the methods that can be used to interact with an MDP through
 * the BatchCore class. It includes methods to run the learning over a dataset and log the progress of the algorithm.
 */
template<class ActionC, class StateC>
class BatchAgent
{
public:
    BatchAgent() :
        gamma(1),
        converged(false)
    {
    }

    /*!
     * This method setup the dataset to be used in the learning process. Must be implemented.
     * Is called by the BatchCore as the first step of the learning process.
     * \param data the dataset to be used for learning
     */
    virtual void init(Dataset<ActionC, StateC>& data, EnvironmentSettings& envSettings) = 0;

    /*!
     * This method implement a step of the learning process trough the dataset. Must be implemented.
     * Is called by the BatchCore until the algorithm converges or the maximum number of iteration is reached.
     */
    virtual void step() = 0;

    /*!
     * This method is used to log agent step informations.
     * Can be overloaded to return information that can be processed by the logger.
     * By default a null pointer is returned, which means that no data will be logged.
     * \return the data to be logged from the agent at the current step.
     */
    virtual AgentOutputData* getAgentOutputData()
    {
        return nullptr;
    }

    /*!
     * This method is used to log agent informations at episode end.
     * Can be overloaded to return information that can be processed by the logger.
     * By default a null pointer is returned, which means that no data will be logged.
     * \return the data to be logged from the agent at the episode end.
     */
    virtual AgentOutputData* getAgentOutputDataEnd()
    {
        return nullptr;
    }

    /*!
     * Getter.
     * \return the policy learned by the agent
     */
    virtual Policy<ActionC, StateC>* getPolicy() = 0;

    /*!
     * This method returns whether the algorithm has converged or not
     * \return the value of the flag converged
     */
    virtual bool hasConverged()
    {
        return converged;
    }

    virtual ~BatchAgent()
    {
    }

protected:
    //! The task discount factor
    double gamma;
    //! flag to signal convergence of the algorithm
    bool converged;
};

}

#endif /* INCLUDE_RELE_ALGORITHMS_BATCH_BATCHAGENT_H_ */
