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

#ifndef INCLUDE_RELE_APPROXIMATORS_REGRESSORS_ENSEMBLE_H_
#define INCLUDE_RELE_APPROXIMATORS_REGRESSORS_ENSEMBLE_H_

#include "rele/approximators/Regressors.h"


namespace ReLe
{

template<class InputC, class OutputC, bool denseOutput = true>
class Ensemble_ : public BatchRegressor_<InputC, OutputC, denseOutput>
{

public:
    Ensemble_(Features_<InputC>& phi,
              unsigned int outputSize = 1)
        : BatchRegressor_<InputC, OutputC>(phi, outputSize)
    {
    }

    virtual OutputC operator()(const InputC& input) override
    {
        if (this->regressors.size() == 0)
            throw std::runtime_error("Empty ensemble evaluated");

        OutputC out = (*this->regressors[0])(input);

        for(unsigned int i = 1; i < this->regressors.size(); i++)
        {
            auto& regressor = *this->regressors[i];
            out += regressor(input);
        }

        return out / static_cast<double>(this->regressors.size());
    }

    virtual void trainFeatures(const BatchData_<OutputC, denseOutput>& featureDataset) override
    {
        for(auto regressor : regressors)
            regressor->trainFeatures(featureDataset);
    }

    virtual double computeJFeatures(const BatchData_<OutputC, denseOutput>& featureDataset) override
    {
        //TODO [IMPORTANT][INTERFACE] implement, probably this method cannot be called by ensemble...
        assert(false);
        return 0;
    }

    BatchRegressor_<InputC, OutputC, denseOutput>& getRegressor(unsigned int index)
    {
        return *regressors[index];
    }

    virtual void writeOnStream(std::ofstream& out) = 0;

    virtual void readFromStream(std::ifstream& in) = 0;

    virtual ~Ensemble_()
    {
        cleanEnsemble();
    }

protected:
    void cleanEnsemble()
    {
        for(auto regressor : regressors)
            if(regressor)
                delete regressor;
    }

protected:
    std::vector<BatchRegressor_<InputC, OutputC, denseOutput>*> regressors; // The regressors ensemble
};

typedef Ensemble_<arma::vec, arma::vec> Ensemble;

}

#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_ENSEMBLE_H_ */
