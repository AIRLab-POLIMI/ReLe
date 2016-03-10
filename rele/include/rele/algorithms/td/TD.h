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

#ifndef INCLUDE_ALGORITHMS_TD_TD_H_
#define INCLUDE_ALGORITHMS_TD_TD_H_

#include "rele/core/Agent.h"
#include "rele/approximators/regressors/others/LinearApproximator.h"
#include "rele/policy/q_policy/ActionValuePolicy.h"
#include "rele/algorithms/step_rules/LearningRate.h"

namespace ReLe
{

class FiniteTDOutput : virtual public AgentOutputData
{
public:
    FiniteTDOutput(double gamma,
                   const std::string& alpha,
                   const std::string& policyName,
                   const hyperparameters_map& policyHPar,
                   const arma::mat& Q);

    virtual void writeData(std::ostream& os) override;
    virtual void writeDecoratedData(std::ostream& os) override;

protected:
    double gamma;
    std::string alpha;
    std::string policyName;
    hyperparameters_map policyHPar;
    arma::mat Q;
};

class LinearTDOutput : virtual public AgentOutputData
{
public:
    LinearTDOutput(double gamma,
                   const std::string& alpha,
                   const std::string& policyName,
                   const hyperparameters_map& policyHPar,
                   const arma::vec Qw);

    virtual void writeData(std::ostream& os) override;
    virtual void writeDecoratedData(std::ostream& os) override;

protected:
    double gamma;
    std::string alpha;
    std::string policyName;
    hyperparameters_map policyHPar;
    arma::vec Qw;
};


class FiniteTD: public Agent<FiniteAction, FiniteState>
{
public:
    FiniteTD(ActionValuePolicy<FiniteState>& policy, LearningRate& alpha);

    virtual void endEpisode() override;

    inline virtual AgentOutputData* getAgentOutputDataEnd() override
    {
        return new FiniteTDOutput(task.gamma, alpha.print(), policy.getPolicyName(),
                                  policy.getPolicyHyperparameters(), Q);
    }

    inline void resetLearningRate()
    {
        alpha.reset();
    }

protected:
    virtual void init() override;

protected:
    //Action-value function
    arma::mat Q;

    //current an previous actions and states
    size_t x;
    unsigned int u;

    //algorithm parameters
    LearningRate& alpha;
    ActionValuePolicy<FiniteState>& policy;

};

class LinearTD : public Agent<FiniteAction, DenseState>
{
public:
    LinearTD(Features& phi, ActionValuePolicy<DenseState>& policy, LearningRateDense& alpha);

    virtual void endEpisode() override;

    inline virtual AgentOutputData* getAgentOutputDataEnd() override
    {
        return new LinearTDOutput(task.gamma, alpha.print(), policy.getPolicyName(),
                                  policy.getPolicyHyperparameters(), Q.getParameters());
    }

    inline void resetLearningRate()
    {
        alpha.reset();
    }

protected:
    virtual void init() override;

protected:
    //Linear action-value function
    LinearApproximator Q;

    //current an previous actions and states
    DenseState x;
    unsigned int u;

    //algorithm parameters
    LearningRateDense& alpha;
    ActionValuePolicy<DenseState>& policy;
};

}//end namespace

#endif /* INCLUDE_ALGORITHMS_TD_TD_H_ */
