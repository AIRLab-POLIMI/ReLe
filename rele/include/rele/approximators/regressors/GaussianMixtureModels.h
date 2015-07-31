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

#ifndef INCLUDE_RELE_APPROXIMATORS_REGRESSORS_GAUSSIANMIXTUREMODELS_H_
#define INCLUDE_RELE_APPROXIMATORS_REGRESSORS_GAUSSIANMIXTUREMODELS_H_

#include "Regressors.h"
#include "ArmadilloPDFs.h"

namespace ReLe
{

template<class InputC>
class GaussianRegressor_ : public ParametricRegressor_<InputC>
{
public:
    GaussianRegressor_(Features_<InputC>& phi) : ParametricRegressor_<InputC>(1), phi(phi)
    {
        unsigned int size = phi.rows();
        sigma = 0.1*arma::eye(size, size);
        mu.set_size(size);
    }

    virtual void setParameters(const arma::vec& params)
    {
        mu = params;
    }

    virtual arma::vec getParameters() const
    {
        return mu;
    }

    virtual unsigned int getParametersSize() const
    {
        return mu.n_elem;
    }

    virtual arma::vec operator() (const InputC& input)
    {
        const arma::vec& value = phi(input);

        arma::vec result(1);
        result(0) = mvnpdf(value, mu, sigma);

        return result;
    }

    virtual arma::vec diff(const arma::vec& input)
    {
        arma::vec g_mean;
        arma::vec g_sigma;
        const arma::vec& value = phi(input);

        mvnpdf(value, mu, sigma, g_mean, g_sigma);

        return g_mean;
    }

    virtual ~GaussianRegressor_()
    {

    }

private:
    Features_<InputC>& phi;
    arma::vec mu;
    arma::mat sigma;

};

typedef GaussianRegressor_<arma::vec> GaussianRegressor;

}


#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_GAUSSIANMIXTUREMODELS_H_ */
