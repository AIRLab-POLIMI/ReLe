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
    DEFINE_FEATURES_TYPES(denseOutput)

public	:
    Autoencoder_(Features_<InputC, denseOutput>& phi, unsigned int outputs)
        : FFNeuralNetwork_<InputC, denseOutput>(phi, outputs, phi.rows()), UnsupervisedBatchRegressor_<InputC, arma::vec, denseOutput>(phi, outputs)
    {
        normalizationF = new MinMaxNormalization<denseOutput>();
        normalizationO = new MinMaxNormalization<denseOutput>();
    }

    virtual arma::vec operator()(const InputC& input) override
    {
        this->forwardComputation(Base::phi(input));
        return this->h[1];
    }

    virtual arma::vec diff(const InputC& input) override
    {
        //TODO [IMPORTANT][INTERFACE] implement. vectorial diff?
        return arma::vec();
    }

    virtual void trainFeatures(const FeaturesCollection& features) override
    {
        //Set normalization
        this->getHyperParameters().normalizationF = normalizationF;
        this->getHyperParameters().normalizationO = normalizationO;

        BatchDataSimple dataset(features, features);
        FFNeuralNetwork_<InputC, denseOutput>::trainFeatures(dataset);
    }

    double computeJFeatures(const arma::mat& features)
    {
        //Run training
        BatchDataSimple data(features, features);
        return FFNeuralNetwork_<InputC, denseOutput>::computeJFeatures(data);
    }

    virtual ~Autoencoder_()
    {

    }

private:
    Normalization<denseOutput>* normalizationF;
    Normalization<denseOutput>* normalizationO;
};

typedef Autoencoder_<arma::vec> Autoencoder;

}

#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_NN_AUTOENCODER_H_ */
