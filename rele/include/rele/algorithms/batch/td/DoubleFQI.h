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

/*
 * Written by: Alessandro Nuara, Carlo D'Eramo
 */

#ifndef INCLUDE_RELE_ALGORITHMS_BATCH_DOUBLEFQI_H_
#define INCLUDE_RELE_ALGORITHMS_BATCH_DOUBLEFQI_H_

#include "rele/algorithms/batch/td/FQI.h"
#include "rele/approximators/Ensemble.h"


namespace ReLe
{

class DoubleFQIEnsemble : public Ensemble
{
public:
    DoubleFQIEnsemble(BatchRegressor& QRegressorA,
                      BatchRegressor& QRegressorB) :
        Ensemble(QRegressorA.getFeatures()),
        currentRegressor(0)
    {
        regressors.push_back(&QRegressorA);
        regressors.push_back(&QRegressorB);
    }

    virtual void trainFeatures(const BatchData& miniBatch) override
    {
        regressors[currentRegressor]->trainFeatures(miniBatch);
    }

    void setCurrentRegressor(unsigned int currentRegressor)
    {
        this->currentRegressor = currentRegressor;
    }

    virtual void writeOnStream(std::ofstream& out) override
    {
    }

    virtual void readFromStream(std::ifstream& in) override
    {
    }

    virtual ~DoubleFQIEnsemble()
    {
        regressors.clear();
    }

protected:
    unsigned int currentRegressor;
};

template<class StateC>
class DoubleFQI: public FQI<StateC>
{
public:
    DoubleFQI(BatchRegressor& QRegressorA,
              BatchRegressor& QRegressorB,
              unsigned int nStates,
              unsigned int nActions,
              double gamma,
              double epsilon,
              bool shuffle = false) :
        FQI<StateC>(QRegressorEnsemble, nStates, nActions, gamma, epsilon),
        QRegressorEnsemble(QRegressorA, QRegressorB),
        shuffle(shuffle)
    {
    }

    void step() override
    {
        arma::uvec allIndexes = arma::conv_to<arma::uvec>::from(
                                    arma::linspace(0, this->nSamples - 1, this->nSamples));
        if(shuffle)
            allIndexes = arma::shuffle(allIndexes);

        std::vector<arma::mat> features;
        std::vector<arma::vec> rewards;
        std::vector<arma::vec> nextStates;
        std::vector<arma::mat> outputs;

        indexes.push_back(allIndexes(arma::span(0, floor(this->nSamples / 2) - 1)));
        indexes.push_back(allIndexes(arma::span(floor(this->nSamples / 2), this->nSamples - 1)));
        for(unsigned int i = 0; i < 2; i++)
        {
            features.push_back(this->features.cols(indexes[i]));
            rewards.push_back(this->rewards(indexes[i]));
            nextStates.push_back(this->nextStates(indexes[i]));
            outputs.push_back(arma::mat(1, indexes[i].n_elem, arma::fill::zeros));

            for(unsigned int j = 0; j < indexes[i].n_elem; j++)
            {
                FiniteState nextState = FiniteState(nextStates[i](j));
                if(this->absorbingStates.count(nextState) == 0 && !this->firstStep)
                {
                    arma::vec Q_xn(this->nActions, arma::fill::zeros);
                    for(unsigned int u = 0; u < this->nActions; u++)
                        Q_xn(u) = arma::as_scalar(
                                      QRegressorEnsemble.getRegressor(i)(nextState, FiniteAction(u)));

                    double qmax = Q_xn.max();
                    arma::uvec maxIndex = find(Q_xn == qmax);
                    unsigned int index = RandomGenerator::sampleUniformInt(0, maxIndex.n_elem - 1);

                    outputs[i](j) = rewards[i](j) + this->gamma * arma::as_scalar(
                                        QRegressorEnsemble.getRegressor(1 - i)(
                                            nextState, FiniteAction(maxIndex(index))));
                }
                else
                    outputs[i](j) = rewards[i](j);
            }
        }

        for(unsigned int i = 0; i < 2; i++)
        {
            QRegressorEnsemble.setCurrentRegressor(i);

            BatchDataSimple featureDataset(features[i], outputs[i]);
            QRegressorEnsemble.trainFeatures(featureDataset);
        }

        this->firstStep = false;

        this->checkCond();

        indexes.clear();
    }

protected:
    DoubleFQIEnsemble QRegressorEnsemble;
    std::vector<arma::uvec> indexes;
    bool shuffle;
};

}

#endif /* INCLUDE_RELE_ALGORITHMS_BATCH_DOUBLEFQI_H_ */
