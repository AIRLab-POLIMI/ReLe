/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo
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

#ifndef INCLUDE_RELE_APPROXIMATORS_REGRESSORS_NN_AUTOENCODER_H_
#define INCLUDE_RELE_APPROXIMATORS_REGRESSORS_NN_AUTOENCODER_H_

#include "FFNeuralNetwork.h"

namespace ReLe
{

template<class InputC, bool denseOutput = true>
class Autoencoder_: public UnsupervisedBatchRegressor_<InputC, arma::vec, denseOutput>,
    public FFNeuralNetwork_<InputC, denseOutput>
{
    USE_UNSUPERVISED_REGRESSOR_MEMBERS(InputC, arma::vec, true)

public	:
    Autoencoder_(Features_<InputC, denseOutput>& phi, unsigned int outputs)
        : FFNeuralNetwork_<InputC, denseOutput>(phi, outputs, phi.rows()), UnsupervisedBatchRegressor_<InputC, arma::vec, denseOutput>(phi, outputs)
    {

    }

    virtual arma::vec operator()(const InputC& input) override
    {
        this->forwardComputation(Base::phi(input));
        return this->h[1];
    }

    virtual arma::vec diff(const InputC& input) override
    {
        //TODO implement
        return arma::vec();
    }

    virtual void trainFeatures(const arma::mat& features) override
    {
        BatchDataFeatures<arma::vec, arma::vec> dataset(features, features);
        FFNeuralNetwork_<InputC>::trainFeatures(dataset);
    }

    virtual double computeJFeatures(const arma::mat& features)
    {
    	BatchDataFeatures<arma::vec, arma::vec> data(features, features);
    	return this->computeJ(data, 0);
    }

    virtual ~Autoencoder_()
    {

    }
};

typedef Autoencoder_<arma::vec> Autoencoder;

}

#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_NN_AUTOENCODER_H_ */
