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


#ifndef INCLUDE_RELE_UTILS_NUMERICALGRADIENTS_H_
#define INCLUDE_RELE_UTILS_NUMERICALGRADIENTS_H_

#include <armadillo>
#include "rele/approximators/Regressors.h"
#include "rele/policy/Policy.h"

namespace ReLe
{

/*!
 * This class implements functions to compute the numerical gradient
 * of functions. The real gradient is approximated with the
 * well-known incremental formula (with a small value of \f$\epsilon\f$), i.e.,
 * \f[\frac{\partial J}{\partial \theta} \approx \frac{J(\theta + \epsilon) - J(\theta - \epsilon)}{2\epsilon}.\f]
 */
class NumericalGradient
{

public:
    /*!
     * Numerical gradient computation for generic functor.
     * \param J the function to be derived
     * \param theta parameters vector
     * \param size the dimensionality of the output
     * \return the numerical gradient
     */
    template<class F>
    static inline arma::mat compute(F J, arma::vec theta, unsigned int size = 1)
    {

        arma::mat numgrad = arma::zeros(size, theta.n_elem);

        arma::vec perturb = arma::zeros(theta.n_elem);
        double e = 1e-5;

        for (unsigned int p = 0; p < theta.n_elem; p++)
        {
            perturb(p) = e;
            // Set perturbation vector
            arma::vec loss1 = J(theta - perturb);
            arma::vec loss2 = J(theta + perturb);
            // Compute Numerical Gradient
            numgrad.col(p) = (loss2 - loss1) / (2*e);
            perturb(p) = 0;
        }

        return numgrad.t();

    }

    /*!
     * Numerical gradient computation for regression functions.
     * \param regressor regression function to be derived
     * \param theta parameters vector
     * \param input input vector of the regressor
     * \return the numerical gradient
     */
    static inline arma::mat compute(ParametricRegressor& regressor,
                                    arma::vec theta, arma::vec& input)
    {
        arma::vec p = regressor.getParameters();
        auto lambda = [&](const arma::vec& par)
        {
            regressor.setParameters(par);
            arma::vec value = regressor(input);
            regressor.setParameters(p);

            return value;
        };

        return compute(lambda, theta, regressor.getOutputSize());
    }

    /*!
     * Numerical gradient computation for policy functions.
     * \param policy policy function to be derived
     * \param theta parameters vector
     * \param state vector of states
     * \param action vector of actions
     * \return the numerical gradient
     */
    template<class ActionC, class StateC>
    static inline arma::mat compute(DifferentiablePolicy<ActionC, StateC>& policy,
                                    arma::vec theta,
                                    typename state_type<StateC>::const_type_ref state,
                                    typename action_type<ActionC>::const_type_ref action)
    {
        arma::vec p = policy.getParameters();

        auto lambda = [&](const arma::vec& par)
        {
            arma::vec value(1);
            policy.setParameters(par);
            value(0) = policy(state, action);
            policy.setParameters(p);

            return value;
        };

        return compute(lambda, theta);
    }

};

}


#endif /* INCLUDE_RELE_UTILS_NUMERICALGRADIENTS_H_ */
