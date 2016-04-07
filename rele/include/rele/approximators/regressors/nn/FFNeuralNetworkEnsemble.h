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

#ifndef INCLUDE_RELE_APPROXIMATORS_REGRESSORS_FFNEURALNETWORKENSEMBLE_H_
#define INCLUDE_RELE_APPROXIMATORS_REGRESSORS_FFNEURALNETWORKENSEMBLE_H_

#include "rele/approximators/regressors/nn/FFNeuralNetwork.h"
#include "rele/approximators/regressors/Ensemble.h"


namespace ReLe
{

template<class InputC>
class FFNeuralNetworkEnsemble_: public Ensemble_<InputC, arma::vec>
{
public:
    FFNeuralNetworkEnsemble_(Features_<InputC>& phi,
                             unsigned int neurons,
                             unsigned int outputSize = 1,
                             unsigned int nRegressors = 50)
        : Ensemble_<InputC, arma::vec>(phi, outputSize)
    {
        initialize(nRegressors, neurons);
    }

    void initialize(unsigned int nRegressors, unsigned int neurons)
    {
        this->cleanEnsemble();
        this->regressors.clear();
        for (unsigned int i = 0; i < nRegressors; i++)
        {
            auto nn = new FFNeuralNetwork_<InputC>(this->phi, neurons, this->outputDimension);
            this->regressors.push_back(nn);
        }
    }

    virtual void writeOnStream(std::ofstream& out)
    {
        //TODO [SERIALIZATION] implement
    }

    virtual void readFromStream(std::ifstream& in)
    {
        //TODO [SERIALIZATION] implement
    }

    virtual ~FFNeuralNetworkEnsemble_()
    {

    }
};

typedef FFNeuralNetworkEnsemble_<arma::vec> FFNeuralNetworkEnsemble;

}

#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_FFNEURALNETWORKENSEMBLE_H_ */
