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

template<bool denseInput = true>
class Autoencoder_: public UnsupervisedBatchRegressor_<arma::vec, denseInput>,
    public FFNeuralNetwork_<denseInput>
{
public:
    DEFINE_FEATURES_TYPES(denseInput)

public	:
    Autoencoder_(unsigned int inputs, unsigned int outputs)
        : FFNeuralNetwork_<denseInput>(inputs, outputs),
		  UnsupervisedBatchRegressor_<InputC, arma::vec, denseInput>(inputs, outputs)
    {
        normalizationF = new MinMaxNormalization<denseInput>();
        normalizationO = new MinMaxNormalization<denseInput>();
    }

    virtual arma::vec operator()(const FeaturesType& input) override
    {
        this->forwardComputation(input);
        return this->h[1];
    }

    virtual arma::vec diff(const FeaturesType& input) override
    {
        //TODO [IMPORTANT][INTERFACE] implement. vectorial diff?
        return arma::vec();
    }

    virtual void train(const FeaturesCollection& features) override
    {
        //Set normalization
        this->getHyperParameters().normalizationF = normalizationF;
        this->getHyperParameters().normalizationO = normalizationO;

        BatchDataSimple dataset(features, features);
        FFNeuralNetwork_<denseInput>::train(dataset);
    }

    double computeJ(const FeaturesCollection& features)
    {
        //Run training
        BatchDataSimple data(features, features);
        return FFNeuralNetwork_<denseInput>::computeJ(data);
    }

    virtual ~Autoencoder_()
    {

    }

private:
    Normalization<denseInput>* normalizationF;
    Normalization<denseInput>* normalizationO;
};

typedef Autoencoder_<> Autoencoder;

}

#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_NN_AUTOENCODER_H_ */
