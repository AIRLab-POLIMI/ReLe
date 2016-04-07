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

#include "rele/policy/Policy.h"
#include "rele/approximators/regressors/others/LinearApproximator.h"
#include <armadillo>

namespace ReLe
{

/**
 * This policy is a linear combination of linear/non linear basis functions
 * \f[ \forall x \in \mathcal{X},\qquad \pi(x) = \sum_{i=1}^{n} w_i \phi_i(x)\f]
 * where \f$n\f$ is the number of parameters and basis functions.
 */
template<class StateC, bool denseFeatures = true>
class DetLinearPolicy: public DifferentiablePolicy<DenseAction, StateC>
{

    using InputC = typename state_type<StateC>::type;

public:

    /**
     * Create an instance of the class using the given features.
     * Note that the weights are initialized
     * by the constructor of the linear approximator
     *
     * \param phi the features \f$\phi(x,u)\f$
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
    std::string getPolicyName() override
    {
        return std::string("DeterministicLinearPolicy");
    }

    std::string printPolicy() override
    {
        std::stringstream ss;
        ss << approximator.getParameters().t();
        return ss.str();
    }

    virtual arma::vec operator()(typename state_type<StateC>::const_type_ref state) override
    {
        return approximator(state);
    }

    virtual double operator()(typename state_type<StateC>::const_type_ref state,
                              const arma::vec& action) override
    {
        arma::vec output = approximator(state);

        DenseAction a(output);

        if (a.isAlmostEqual(action))
        {
            return 1.0;
        }
        return 0.0;
    }

    virtual DetLinearPolicy<StateC, denseFeatures>* clone() override
    {
        return new DetLinearPolicy<StateC, denseFeatures>(*this);
    }

    // ParametricPolicy interface
public:
    virtual inline arma::vec getParameters() const override
    {
        return approximator.getParameters();
    }
    virtual inline const unsigned int getParametersSize() const override
    {
        return approximator.getParametersSize();
    }
    virtual inline void setParameters(const arma::vec& w) override
    {
        approximator.setParameters(w);
    }

    // DifferentiablePolicy interface
public:
    //TODO [IMPORTANT][INTERFACE] is this semantically correct? is not the diff of a probability.... it's a different one
    arma::vec diff(typename state_type<StateC>::const_type_ref state, const arma::vec& action) override
    {
        //FIXME [IMPORTANT] this only works when considering a scalar
        return approximator.diff(state);
    }

    arma::vec difflog(typename state_type<StateC>::const_type_ref state, const arma::vec& action) override
    {
        //FIXME [IMPORTANT] this only works when considering a scalar
        return approximator.diff(state) / approximator(state);
    }

    arma::mat diff2log(typename state_type<StateC>::const_type_ref state, const arma::vec& action) override
    {
        //TODO [IMPORTANT] this only works when considering a scalar
        arma::mat phi = approximator.diff(state);
        double value = arma::as_scalar(approximator(state));
        return phi*phi.t()/(value*value);
    }

protected:
    LinearApproximator_<InputC, denseFeatures> approximator;

};

} //end namespace

#endif //LINEARPOLICY_H_
