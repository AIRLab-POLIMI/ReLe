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

/*!
 * This class implements the output data for all Finite TD algorithms.
 * All Finite TD algorithms should use, or extend this class
 */
class FiniteTDOutput : virtual public AgentOutputData
{
public:
    /*!
     * Constructor.
     * \param gamma the discount factor
     * \param alpha a string describing the learning rate
     * \param policyName the name of the policy used
     * \param policyHPar the map of the hyperparameters of the policy
     * \param Q the Q-table
     */
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

/*!
 * This class implements the output data for linear approximated state TD algorithms.
 * All linear linear approximated state TD algorithms should use, or extend this class
 */
class LinearTDOutput : virtual public AgentOutputData
{
public:
    /*!
     * Constructor.
     * \param gamma the discount factor
     * \param alpha a string describing the learning rate
     * \param policyName the name of the policy used
     * \param policyHPar the map of the hyperparameters of the policy
     * \param Qw the weights of the linear approximator of the Q-Function
     */
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

/*!
 * This interface is the basic interface for all Finite TD algorithms.
 * A finite TD algorithm is a Q-table based algorithm,
 * this means that both action and state space are finite.
 */
class FiniteTD: public Agent<FiniteAction, FiniteState>
{
public:
    /*!
     * Constructor.
     * \param policy the policy to be used by the algorithm
     * \param alpha the learning rate to be used by the algorithm
     */
    FiniteTD(ActionValuePolicy<FiniteState>& policy, LearningRate& alpha);

    virtual void endEpisode() override;

    inline virtual AgentOutputData* getAgentOutputDataEnd() override
    {
        return new FiniteTDOutput(task.gamma, alpha.print(), policy.getPolicyName(),
                                  policy.getPolicyHyperparameters(), Q);
    }

    /*!
     * This method resets the algorithm learning rate.
     */
    inline void resetLearningRate()
    {
        alpha.reset();
    }

protected:
    /*!
     * Implementation of the init method.
     * Initializes the Q-table and setup the policy to use the Q-Table.
     */
    virtual void init() override;

protected:
    //! Action-value function
    arma::mat Q;
    //! previous state
    size_t x;
    //! previous action
    unsigned int u;
    //! learning rate
    LearningRate& alpha;
    //! algorithm policy
    ActionValuePolicy<FiniteState>& policy;

};

/*!
 * This interface is the basic interface for all linear TD algorithms.
 * A linear TD algorithm is an algorithm using finite action spaces
 * and dense state spaces where the approximation of the action-values
 * is performed with linear approximation.
 */
class LinearTD : public Agent<FiniteAction, DenseState>
{
public:
    /*!
     * Constructor.
     * \param phi the features to be used for linear approximation of the state space
     * \param policy the policy to be used by the algorithm
     * \param alpha the learning rate to be used by the algorithm
     */
    LinearTD(Features& phi, ActionValuePolicy<DenseState>& policy, LearningRateDense& alpha);

    virtual void endEpisode() override;

    inline virtual AgentOutputData* getAgentOutputDataEnd() override
    {
        return new LinearTDOutput(task.gamma, alpha.print(), policy.getPolicyName(),
                                  policy.getPolicyHyperparameters(), Q.getParameters());
    }

    /*!
     * This method resets the algorithm learning rate.
     */
    inline void resetLearningRate()
    {
        alpha.reset();
    }

protected:
    /*!
     * Implementation of the init method.
     * Initializes the Q-function approximator, and setup the policy to use the Q-Table.
     */
    virtual void init() override;

protected:
    //! Linear approximated action-value function
    LinearApproximator Q;
    //! previous state
    DenseState x;
    //! previous action
    unsigned int u;
    //! learning rate
    LearningRateDense& alpha;
    //! algorithm policy
    ActionValuePolicy<DenseState>& policy;
};

}//end namespace

#endif /* INCLUDE_ALGORITHMS_TD_TD_H_ */
