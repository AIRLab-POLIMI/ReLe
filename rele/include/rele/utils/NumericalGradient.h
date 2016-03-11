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


#ifndef INCLUDE_RELE_UTILS_NUMERICALGRANDIENTS_H_
#define INCLUDE_RELE_UTILS_NUMERICALGRANDIENTS_H_

#include <armadillo>
#include "rele/approximators/Regressors.h"
#include "rele/policy/Policy.h"

namespace ReLe
{

class NumericalGradient
{

public:
    /**
     * @brief Numerical gradient computation for generic functor
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

    /**
     * @brief Numerical gradient computation for regressor
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

    /**
     * @brief Numerical gradient computation for policies
     */
    template<class StateC>
    static inline arma::mat compute(DifferentiablePolicy<DenseAction, StateC>& policy,
                                    arma::vec theta,
                                    typename state_type<StateC>::const_type_ref input,
                                    const arma::vec& output)
    {
        arma::vec p = policy.getParameters();

        auto lambda = [&](const arma::vec& par)
        {
            arma::vec value(1);
            policy.setParameters(par);
            value(0) = policy(input, output);
            policy.setParameters(p);

            return value;
        };

        return compute(lambda, theta);
    }

    /**
     * @brief Numerical gradient computation for policies
     */
    template<class StateC>
    static inline arma::mat compute(DifferentiablePolicy<FiniteAction, StateC>& policy,
                                    arma::vec theta,
                                    typename state_type<StateC>::const_type_ref input,
                                    const FiniteAction& output)
    {
        arma::vec p = policy.getParameters();

        auto lambda = [&](const arma::vec& par)
        {
            arma::vec value(1);
            policy.setParameters(par);
            value(0) = policy(input, output);
            policy.setParameters(p);

            return value;
        };

        return compute(lambda, theta);
    }

};

}


#endif /* INCLUDE_RELE_UTILS_NUMERICALGRANDIENTS_H_ */
