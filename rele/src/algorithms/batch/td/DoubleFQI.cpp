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

#include "rele/algorithms/batch/td/DoubleFQI.h"

namespace ReLe
{
DoubleFQIEnsemble::DoubleFQIEnsemble(BatchRegressor& QRegressorA,
                                     BatchRegressor& QRegressorB) :
    Ensemble(QRegressorA.getFeatures())
{
    regressors.push_back(&QRegressorA);
    regressors.push_back(&QRegressorB);
}

void DoubleFQIEnsemble::writeOnStream(std::ofstream& out)
{
}

void DoubleFQIEnsemble::readFromStream(std::ifstream& in)
{
}

DoubleFQIEnsemble::~DoubleFQIEnsemble()
{
    regressors.clear();
}

DoubleFQI::DoubleFQI(BatchRegressor& QRegressorA,
                     BatchRegressor& QRegressorB,
                     unsigned int nActions,
                     double epsilon,
                     bool shuffle) :
    FQI(QRegressorEnsemble, nActions, epsilon),
    QRegressorEnsemble(QRegressorA, QRegressorB),
    shuffle(shuffle)
{
}

void DoubleFQI::step()
{
    arma::uvec allIndexes = arma::conv_to<arma::uvec>::from(
                                arma::linspace(0, this->nSamples - 1, this->nSamples));
    if(shuffle)
        allIndexes = arma::shuffle(allIndexes);

    indexes.push_back(allIndexes(arma::span(0, floor(this->nSamples / 2) - 1)));
    indexes.push_back(allIndexes(arma::span(floor(this->nSamples / 2), this->nSamples - 1)));
    for(unsigned int i = 0; i < 2; i++)
    {
        arma::mat features = this->features.cols(indexes[i]);
        arma::vec rewards = this->rewards(indexes[i]);
        arma::mat nextStates = this->nextStates.cols(indexes[i]);
        arma::mat outputs(1, indexes[i].n_elem, arma::fill::zeros);

        for(unsigned int j = 0; j < indexes[i].n_elem; j++)
        {
            if(this->absorbingStates.count(indexes[i][j]) == 0 && !this->firstStep)
            {
                arma::vec Q_xn(this->nActions, arma::fill::zeros);
                for(unsigned int u = 0; u < this->nActions; u++)
                    Q_xn(u) = arma::as_scalar(
                                  QRegressorEnsemble.getRegressor(i)(
                                      nextStates.col(j), FiniteAction(u)));

                double qmax = Q_xn.max();
                arma::uvec maxIndex = find(Q_xn == qmax);
                unsigned int index = RandomGenerator::sampleUniformInt(0,
                                     maxIndex.n_elem - 1);

                outputs(j) = rewards(j) + this->gamma * arma::as_scalar(
                                 QRegressorEnsemble.getRegressor(1 - i)(
                                     nextStates.col(j), FiniteAction(maxIndex(index))));
            }
            else
                outputs(j) = rewards(j);
        }

        BatchDataSimple featureDataset(features, outputs);
        QRegressorEnsemble.getRegressor(i).trainFeatures(featureDataset);
    }

    this->firstStep = false;

    this->checkCond();

    indexes.clear();
}

}
