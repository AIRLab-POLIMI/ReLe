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

#ifndef PORTFOLIONORMALPOLICY_H
#define PORTFOLIONORMALPOLICY_H

#include "rele/policy/Policy.h"
#include "rele/approximators/regressors/others/LinearApproximator.h"
#include "rele/utils/ArmadilloPDFs.h"

namespace ReLe
{

///////////////////////////////////////////////////////////////////////////////////////
/// PORTFOLIO NORMAL POLICY
///////////////////////////////////////////////////////////////////////////////////////

/**
 * Univariate normal policy with fixed standard deviation
 */
class PortfolioNormalPolicy: public DifferentiablePolicy<FiniteAction, DenseState>
{
public:
    PortfolioNormalPolicy(const double& epsilon, Features& phi) :
        epsilon(epsilon),
        approximator(phi)
    {
    }

    virtual ~PortfolioNormalPolicy()
    {

    }

public:

    virtual double operator()(const arma::vec& state,
                              typename action_type<FiniteAction>::const_type_ref action) override;

    virtual unsigned int operator()(const arma::vec& state) override;


    virtual inline std::string getPolicyName() override
    {
        return "PortfolioNormalPolicy";
    }

    virtual inline std::string printPolicy() override
    {
        return "";
    }

    virtual PortfolioNormalPolicy* clone() override
    {
        return new  PortfolioNormalPolicy(*this);
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
    virtual arma::vec diff(const arma::vec& state,
                           typename action_type<FiniteAction>::const_type_ref action) override;

    virtual arma::vec difflog(const arma::vec& state,
                              typename action_type<FiniteAction>::const_type_ref action) override;

    virtual arma::mat diff2log(const arma::vec& state,
                               typename action_type<FiniteAction>::const_type_ref action) override;

protected:
    double epsilon;
    LinearApproximator approximator;
};

} // end namespace ReLe
#endif // PORTFOLIONORMALPOLICY_H
