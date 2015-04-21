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
template<class StateC>
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
    DetLinearPolicy(Features& phi) :
        approximator(phi)
    {
    }

    virtual ~DetLinearPolicy()
    {
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
        std::stringstream ss;
        ss << approximator.getParameters().t();
        return ss.str();
    }

    virtual arma::vec operator()(const arma::vec& state)
    {
        return approximator(state);
    }

    virtual double operator()(const arma::vec& state, const arma::vec& action)
    {
        arma::vec output = (*this)(state);

        //TODO CONTROLLARE ASSEGNAMENTO
        DenseAction a;
        a.copy_vec(output);
        if (a.isAlmostEqual(action))
        {
            return 1.0;
        }
        return 0.0;
    }

    virtual DetLinearPolicy<StateC>* clone()
    {
        return new DetLinearPolicy<StateC>(*this);
    }

    // ParametricPolicy interface
public:
    virtual inline arma::vec getParameters() const
    {
        return approximator.getParameters();
    }
    virtual inline const unsigned int getParametersSize() const
    {
        return approximator.getParametersSize();
    }
    virtual inline void setParameters(arma::vec& w)
    {
        approximator.setParameters(w);
    }

    // DifferentiablePolicy interface
public:
    arma::vec diff(const arma::vec& state, const arma::vec& action)
    {
        return approximator.diff(state);
    }

    arma::vec difflog(const arma::vec& state, const arma::vec& action)
    {
        //TODO ???

        return arma::vec();
    }

protected:
    LinearApproximator approximator;

};

#undef DETLINPOL_NAME

} //end namespace

#endif //LINEARPOLICY_H_
