/*
*  rele,
*
*
*  Copyright (C) 2015 Davide Tateo & Matteo Pirotta
*  Versione 1.0
*
*  This file is part of rele.
*
*  rele is free software: you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation, either version 3 of the License, or
*  (at your option) any later version.
*
*  rele is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with rele.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "rele/core/BatchAgent.h"
#include "rele/policy/q_policy/e_Greedy.h"

#ifndef INCLUDE_RELE_ALGORITHMS_BATCH_TD_BATCHTDAGENT_H_
#define INCLUDE_RELE_ALGORITHMS_BATCH_TD_BATCHTDAGENT_H_

namespace ReLe
{
/*!
 * The BatchTDAgent is the basic interface of all TD batch agents.
 * All batch TD algorithms should extend this abstract class.
 * This class provides method to derive the policy from the
 * regressor.
 */
template<class StateC>
class BatchTDAgent : public BatchAgent<FiniteAction, StateC>
{
public:
    /*!
     * Constructor
     * \param QRegressor the regressor
     * \param nActions the number of actions
     */
    BatchTDAgent(BatchRegressor& QRegressor, unsigned int nActions) :
        QRegressor(QRegressor),
        nActions(nActions),
        policy(nullptr)
    {
    }

    /*!
     * Getter.
     * \return the policy learned by the agent
     */
    virtual Policy<FiniteAction, StateC>* getPolicy() override
    {
        policy->setQ(&QRegressor);
        policy->setNactions(nActions);

        return policy;
    }

    /*!
     * Setter.
     * This function is used to set the type of policy to use
     * with the regressor.
     * \param policy the type of policy to use
     */
    inline void setPolicy(ActionValuePolicy<StateC>* policy)
    {
        this->policy = policy;
    }

    /*
     * Getter.
     * \return the regressor
     */
    inline BatchRegressor& getQRegressor()
    {
        return QRegressor;
    }

    virtual ~BatchTDAgent()
    {
    }

protected:
    BatchRegressor& QRegressor;
    unsigned int nActions;
    ActionValuePolicy<StateC>* policy;
};

}

#endif /* INCLUDE_RELE_ALGORITHMS_BATCH_TD_BATCHTDAGENT_H_ */
