/*
 * rele_ros,
 *
 *
 * Copyright (C) 2017 Davide Tateo
 * Versione 1.0
 *
 * This file is part of rele_ros.
 *
 * rele_ros is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * rele_ros is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with rele_ros.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef TEST_EMOTIONS_COMPRESSEDPOLICY_H_
#define TEST_EMOTIONS_COMPRESSEDPOLICY_H_

#include "rele/policy/Policy.h"
#include "rele/approximators/regressors/others/LinearApproximator.h"
#include "rele/approximators/regressors/nn/Autoencoder.h"
#include <armadillo>

namespace ReLe_ROS
{

class CompressedPolicy: public ReLe::ParametricPolicy<ReLe::DenseAction, ReLe::DenseState>
{

public:
    CompressedPolicy(ReLe::Features& phi, ReLe::Autoencoder& decoder);

    virtual ~CompressedPolicy();

    // Policy interface
public:
    std::string getPolicyName() override;

    std::string printPolicy() override;

    virtual arma::vec operator()(const arma::vec& state) override;

    virtual double operator()(const arma::vec& state,
                              const arma::vec& action) override;

    virtual CompressedPolicy* clone() override;

    // ParametricPolicy interface
public:
    virtual arma::vec getParameters() const override;
    virtual const unsigned int getParametersSize() const override;
    virtual void setParameters(const arma::vec& w) override;


protected:
    ReLe::LinearApproximator approximator;
    ReLe::Autoencoder& decoder;

};

} //end namespace



#endif /* TEST_EMOTIONS_COMPRESSEDPOLICY_H_ */
