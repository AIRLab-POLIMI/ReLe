/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo & Matteo Pirotta & Marcello Restelli
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

#ifndef POLICYEVALAGENT_H_
#define POLICYEVALAGENT_H_

#include "rele/core/Agent.h"
#include "rele/policy/Policy.h"
#include "rele/statistics/Distribution.h"
#include "rele/core/BasicFunctions.h"

namespace ReLe
{

/*!
 * This class implements a fake agent, that cannot be used for learning but only for evaluating a policy
 * over an MDP.
 * This class implements the agent interface, however the methods that implement learning should never be called.
 */
template<class ActionC, class StateC>
class PolicyEvalAgent: public Agent<ActionC, StateC>
{
public:
    /*!
     * Constructor
     * \param policy the policy to be used in evaluation
     */
    PolicyEvalAgent(Policy<ActionC, StateC>& policy): policy(policy)
    {

    }

    virtual ~PolicyEvalAgent()
    {
    }

public:
    virtual void initTestEpisode() override
    {
    }

    /*!
     * This method should never be called. Throws an exception if used, as policy eval agent
     * cannot be used for learning.
     */
    virtual void initEpisode(const StateC &state, ActionC &action) override
    {
        throw std::runtime_error("PolicyEvalAgent cannot be used for learning!");
    }

    void sampleAction(const StateC &state, ActionC &action) override
    {
        typename action_type<ActionC>::type_ref u = action;
        u = policy(state);
    }

    /*!
     * This method should never be called. Throws an exception if used, as policy eval agent
     * cannot be used for learning.
     */
    void step(const Reward &reward, const StateC &nextState, ActionC &action) override
    {
        throw std::runtime_error("PolicyEvalAgent cannot be used for learning!");
    }

    /*!
     * This method should never be called. Throws an exception if used, as policy eval agent
     * cannot be used for learning.
     */
    void endEpisode(const Reward& reward) override
    {
        throw std::runtime_error("PolicyEvalAgent cannot be used for learning!");
    }

    /*!
     * This method should never be called. Throws an exception if used, as policy eval agent
     * cannot be used for learning.
     */
    void endEpisode() override
    {
        throw std::runtime_error("PolicyEvalAgent cannot be used for learning!");
    }

private:
    Policy<ActionC, StateC>& policy;

};


/*!
 * This class implements a fake agent, that cannot be used for learning but only for evaluating a
 * distribution of parametric policies over an MDP.
 */
template<class ActionC, class StateC>
class PolicyEvalDistribution : public PolicyEvalAgent<ActionC, StateC>
{
public:

    /*!
     * Constructor
     * \param dist the distribution of the parameters of the policies
     * \param policy the family of parametric policies to be used
     */
    PolicyEvalDistribution(Distribution& dist, ParametricPolicy<ActionC, StateC>& policy)
        : PolicyEvalAgent<ActionC, StateC>(policy), policy(policy), dist(dist)
    {

    }

    virtual void initTestEpisode() override
    {
        //obtain new parameters
        arma::vec new_params = dist();

        //Save them in the history
        params_history.push_back(new_params);

        //set to policy
        policy.setParameters(new_params);
    }

    /*!
     * This method can be used to return the parameters used during the test runs
     * \return a matrix  paramsN \f$\times\f$ episodeN
     */
    arma::mat getParams()
    {
        arma::mat params(policy.getParametersSize(), params_history.size());

        for(int i = 0; i < params.n_cols; i++)
        {
            params.col(i) = params_history[i];
        }

        return params;
    }


    virtual ~PolicyEvalDistribution()
    {

    }

private:
    ParametricPolicy<ActionC, StateC>& policy;
    Distribution& dist;
    std::vector<arma::vec> params_history;
};

} //end namespace

#endif //POLICYEVALAGENT_H_
