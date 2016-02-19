/*
 * rele,
 *
 *
 * Copyright (C) 2016 Davide Tateo
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

#ifndef INCLUDE_RELE_IRL_UTILS_HESSIANCALCULATOR_H_
#define INCLUDE_RELE_IRL_UTILS_HESSIANCALCULATOR_H_

#include "rele/approximators/Features.h"
#include "rele/policy/Policy.h"
#include "rele/core/Transition.h"

namespace ReLe
{

template<class ActionC, class StateC>
class HessianCalculator
{
public:
    HessianCalculator(Features& phi,
                      Dataset<ActionC,StateC>& data,
                      DifferentiablePolicy<ActionC,StateC>& policy,
                      double gamma) : phi(phi), data(data), policy(policy), gamma(gamma)
    {
        computed = false;
    }

    arma::mat computeHessian(const arma::vec& w)
    {
        compute();

        arma::mat H(Hdiff.n_rows, Hdiff.n_cols, arma::fill::zeros);

        for(unsigned int i = 0; i < Hdiff.n_slices; i++)
        {
            H += Hdiff.slice(i)*w(i);
        }

        return H;

    }

    arma::cube getHessianDiff()
    {
        compute();
        return Hdiff;
    }

    virtual ~HessianCalculator()
    {

    }

protected:
    virtual arma::cube computeHessianDiff() = 0;


    arma::mat computeLocalG(Transition<ActionC,StateC>& tr)
    {
        arma::vec logDiff = policy.difflog(tr.x, tr.u);
        arma::mat logDiff2 = policy.diff2log(tr.x, tr.u);
        return logDiff2 + logDiff*logDiff.t();
    }

    arma::mat computeG(Episode<ActionC,StateC>& episode)
    {
        int dp  = policy.getParametersSize();
        int nbSteps = episode.size();
        arma::mat G(dp, dp, arma::fill::zeros);

        //iterate the episode
        for (int t = 0; t < nbSteps; ++t)
        {
            Transition<ActionC, StateC>& tr = episode[t];
            G += computeLocalG(tr);
        }

        return G;
    }

private:
    void compute()
    {
        if(!computed)
        {
            Hdiff = computeHessianDiff();
            computed = true;
        }
    }

protected:
    Features& phi;
    Dataset<ActionC,StateC>& data;
    DifferentiablePolicy<ActionC,StateC>& policy;
    double gamma;

private:
    arma::cube Hdiff;
    bool computed;
};

}

#define USE_HESSIAN_CALCULATOR_MEMBERS(ActionC, StateC) \
			typedef HessianCalculator<ActionC,StateC> Base; \
			using Base::phi; \
			using Base::data; \
			using Base::policy; \
			using Base::gamma;


#endif /* INCLUDE_RELE_IRL_UTILS_HESSIANCALCULATOR_H_ */
