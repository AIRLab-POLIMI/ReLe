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

#ifndef LINEARPOLICY_H_
#define LINEARPOLICY_H_

#include "Policy.h"
#include "LinearApproximator.h"
#include <armadillo>

#define DETLINPOL_NAME "DeterministicLinearPolicy"

namespace ReLe
{


/**
 * @brief Deterministic policy obtained as linear combination of basis functions.
 * This policy is a linear combination of linear/non linear basis functions
 * \f[ \forall s \in S,\qquad \pi(s) = \sum_{i=1}^{n} w_i \phi_i(s)\f]
 * where \f$n\f$ is the number of parameters and basis functions.
 */
template <class StateC>
class DetLinearPolicy: public DifferentiablePolicy<DenseAction, StateC>
{

public:

    /**
     * Create an instance of the class using the given projector.
     * Note that the weights of the mean approximation are not
     * changed, i.e., the initial weights are specified by the
     * instance of the linear projector received as parameter.
     *
     * @brief The constructor.
     * @param projector The linear projector
     */
    DetLinearPolicy(LinearApproximator* projector)
        : approximator(projector), clearRegressorOnExit(false)
    { }

    virtual ~DetLinearPolicy()
    {
        if (clearRegressorOnExit == true)
        {
            delete approximator;
        }
    }

    // Policy interface
public:
    std::string getPolicyName()
    {
        return std::string(DETLINPOL_NAME);
    }

    std::string getPolicyHyperparameters()
    {
        return std::string("");
    }

    std::string printPolicy()
    {
        return std::string("");
    }

    virtual typename action_type<DenseAction>::type operator() (
        typename state_type<StateC>::const_type state)
    {
        //TODO CONTROLLARE ASSEGNAMENTO
        arma::vec output = (*approximator)(state);
        todoAction.copy_vec(output);
        return todoAction;
    }

    virtual double operator() (
        typename state_type<StateC>::const_type state,
        typename action_type<DenseAction>::const_type action)
    {
        arma::vec output = this->operator ()(state);

        //TODO CONTROLLARE ASSEGNAMENTO
        DenseAction a;
        a.copy_vec(output);
        if (a.isAlmostEqual(action))
        {
            return 1.0;
        }
        return 0.0;
    }

    // ParametricPolicy interface
public:
    virtual inline const arma::vec &getParameters() const
    {
        return approximator->getParameters();
    }
    virtual inline const unsigned int getParametersSize() const
    {
        return approximator->getParameters().n_elem;
    }
    virtual inline void setParameters(arma::vec &w)
    {
        approximator->setParameters(w);
    }

    // DifferentiablePolicy interface
public:
    arma::vec diff(
        typename state_type<StateC>::const_type state,
        typename action_type<DenseAction>::const_type action)
    {
        return approximator->diff(state);
    }

    arma::vec difflog(
        typename state_type<StateC>::const_type state,
        typename action_type<DenseAction>::const_type action)
    {
        //TODO ???

        return arma::vec();
    }

    inline void clearRegressor(bool clear)
    {
        clearRegressorOnExit = clear;
    }


protected:
    LinearApproximator* approximator;
    bool clearRegressorOnExit;
    DenseAction todoAction;
};

#undef DETLINPOL_NAME

}//end namespace

#endif //LINEARPOLICY_H_
