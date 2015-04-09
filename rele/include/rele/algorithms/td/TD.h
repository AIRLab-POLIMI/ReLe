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

#include "Agent.h"
#include <armadillo>
#include "LinearApproximator.h"
#include "q_policy/e_Greedy.h"

namespace ReLe
{

class FiniteTDOutput : public AgentOutputData
{
public:
    FiniteTDOutput(double gamma,
                   double alpha,
                   std::string policyName,
                   std::string policyHPar,
                   arma::mat Q);

    virtual void writeData(std::ostream& os);
    virtual void writeDecoratedData(std::ostream& os);

protected:
    double gamma;
    double alpha;
    std::string policyName;
    std::string policyHPar;
    arma::mat Q;
};

class LinearTDOutput : public AgentOutputData
{
public:
    LinearTDOutput(double gamma,
                   double alpha,
                   std::string policyName,
                   std::string policyHPar,
                   arma::vec Qw);

    virtual void writeData(std::ostream& os);
    virtual void writeDecoratedData(std::ostream& os);

protected:
    double gamma;
    double alpha;
    std::string policyName;
    std::string policyHPar;
    arma::vec Qw;
};


class FiniteTD: public Agent<FiniteAction, FiniteState>
{
public:
    FiniteTD(ActionValuePolicy<FiniteState>& policy);

    virtual void endEpisode();

    inline virtual AgentOutputData* getAgentOutputDataEnd()
    {
        return new FiniteTDOutput(task.gamma, alpha, policy.getPolicyName(),
                                  policy.getPolicyHyperparameters(), Q);
    }

    inline void setAlpha(double alpha)
    {
        this->alpha = alpha;
    }

protected:
    virtual void init();

protected:
    //Action-value function
    arma::mat Q;

    //current an previous actions and states
    size_t x;
    unsigned int u;

    //algorithm parameters
    double alpha;
    ActionValuePolicy<FiniteState>& policy;

};

class LinearTD : public Agent<FiniteAction, DenseState>
{
public:
    LinearTD(ActionValuePolicy<DenseState>& policy, LinearApproximator& la);

    virtual void endEpisode();

    inline virtual AgentOutputData* getAgentOutputDataEnd()
    {
        return new LinearTDOutput(task.gamma, alpha, policy.getPolicyName(),
                                  policy.getPolicyHyperparameters(), Q.getParameters());
    }

    inline void setAlpha(double alpha)
    {
        this->alpha = alpha;
    }

protected:
    virtual void init();

protected:
    //Linear action-value function
    LinearApproximator Q;

    //current an previous actions and states
    DenseState x;
    unsigned int u;

    //algorithm parameters
    double alpha;
    ActionValuePolicy<DenseState>& policy;
};

}//end namespace

#endif /* INCLUDE_ALGORITHMS_TD_TD_H_ */
