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

#ifndef LSPI_H_
#define LSPI_H_

#include "rele/algorithms/batch/td/LSTDQ.h"
#include "rele/utils/RandomGenerator.h"

namespace ReLe
{

/*!
 * This class implements the Least-Squares Policy Iteration (LSPI) algorithm.
 * This algorithm is an off-policy batch algorithm that exploits
 * the action-values approximation done by the LSTDQ algorithm
 * to form an approximate policy-iteration algorithm.
 *
 * References
 * =========
 *
 * [Lagoudakis, Parr. Least-Squares Policy Iteration](http://jmlr.csail.mit.edu/papers/volume4/lagoudakis03a/lagoudakis03a.ps)
 */
template<class ActionC>
class LSPI : public BatchAgent<ActionC, DenseState>
{
public:
    /*!
     * Constructor.
     * \param data the dataset
     * \param policy the policy
     * \param phi the features to be used for approximation
     * \param epsilon coefficient used to check whether to stop the training
     */
    LSPI(e_GreedyApproximate& policy,
         Features_<arma::vec>& phi, double epsilon) :
        oldWeights(arma::vec(phi.rows(), arma::fill::zeros)),
        policy(policy),
        phi(phi),
        critic(nullptr),
        epsilon(epsilon),
        firstStep(true)
    {
    }

    virtual void init(Dataset<ActionC, DenseState>& data, EnvironmentSettings& envSettings) override
    {
        critic = new LSTDQ<ActionC>(data, policy, phi, envSettings.gamma);
        firstStep = true;
        this->converged = false;
    }

    virtual void step() override
    {
        //Evaluate the current policy (and implicitly improve)
        arma::vec QWeights = critic->run(firstStep);
        critic->getQ().setParameters(QWeights);

        //check if termination condition has been reached
        if(!firstStep)
            checkCond(QWeights);

        //save old weights
        oldWeights = QWeights;

        //set first step variable
        firstStep = false;
    }

    /*!
     * Check whether the stop condition is satisfied.
     * \param QWeights the current weights
     */
    virtual void checkCond(const arma::vec& QWeights)
    {
        static int i = 0;
        //Compute the distance between the current and the previous policy
        double LMAXnorm = arma::norm(QWeights - oldWeights, "inf");
        double L2norm   = arma::norm(QWeights - oldWeights, 2);
        double distance = L2norm;

        if(distance < epsilon)
            this->converged = true;

        std::cout << i++ << " " << distance << std::endl;
    }

    virtual Policy<ActionC, DenseState>* getPolicy() override
    {
        //TODO [INTERFACE] fix interface implementation for batch methods...
        return nullptr;
    }

    virtual ~LSPI()
    {
        delete critic;
    }

protected:
    LSTDQ<ActionC>* critic;
    Features_<arma::vec>& phi;
    e_GreedyApproximate& policy;
    arma::vec oldWeights;
    double epsilon;
    bool firstStep;
};

}
#endif //LSPI_H_
