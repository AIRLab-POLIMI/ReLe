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
        currentRegressor = 1 - currentRegressor;
    }

    virtual void writeOnStream(std::ofstream& out) override
    {
        // TODO: Implement
    }

    virtual void readFromStream(std::ifstream& in) override
    {
        // TODO: Implement
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

    /* This class implements the Double FQI algorithm. As a batch algorithm, it takes
     * a dataset of (s, a, r, s') transitions, together with the regressors that
     * it is used to approximate the target distribution of Q values.
     */
    DoubleFQI(BatchRegressor& QRegressorA,
              BatchRegressor& QRegressorB,
              unsigned int nStates,
              unsigned int nActions,
              double gamma,
              bool shuffle = false) :
        FQI<StateC>(QRegressorEnsemble, nStates, nActions, gamma, 2),
        QRegressorEnsemble(QRegressorA, QRegressorB),
        shuffle(shuffle)
    {
    }

    void step() override
    {
        std::vector<arma::mat> outputs;
        std::vector<arma::vec> nextStatesMiniBatch;

        auto&& miniBatches = this->featureDatasetStart->getNMiniBatches(this->nMiniBatches);

        for(unsigned int i = 0; i < this->nMiniBatches; i++)
        {
            this->QRegressor.trainFeatures(*miniBatches[i]);
            nextStatesMiniBatch.push_back(this->nextStates(miniBatches[i]->getIndexes()));
            outputs.push_back(miniBatches[i]->getOutputs());
        }

        if(shuffle)
            for(unsigned int i = 0; i < this->nMiniBatches; i++)
            {
                miniBatches[i]->shuffle();
                nextStatesMiniBatch[i] = this->nextStates(miniBatches[i]->getIndexes());
            }

        std::vector<arma::mat> inputs;

        for(unsigned int miniBatchIndex = 0; miniBatchIndex < this->nMiniBatches; miniBatchIndex++)
        {
            arma::mat miniBatchInput = miniBatches[miniBatchIndex]->getFeatures();
            arma::mat miniBatchRewards = miniBatches[miniBatchIndex]->getOutputs();

            for(unsigned int i = 0; i < miniBatchInput.n_cols; i++)
            {
                /* In order to be able to fill the output vector (i.e. regressor
                 * target values), we need to compute the Q values for each
                 * s' sample in the dataset and for each action in the
                 * set of actions of the problem. Recalling the fact that the values
                 * are zero for each action in an absorbing state, we check if s' is
                 * absorbing and, if it is, we leave the Q-values fixed to zero.
                 */
                FiniteState nextState = FiniteState(nextStatesMiniBatch[miniBatchIndex](i));
                if(this->absorbingStates.count(nextState) == 0)
                {
                    arma::vec Q_xn(this->nActions, arma::fill::zeros);
                    for(unsigned int u = 0; u < this->nActions; u++)
                        Q_xn(u) = arma::as_scalar(
                                      QRegressorEnsemble.getRegressor(miniBatchIndex)(nextState, FiniteAction(u)));

                    double qmax = Q_xn.max();
                    arma::uvec maxIndex = find(Q_xn == qmax);
                    unsigned int index = RandomGenerator::sampleUniformInt(0, maxIndex.n_elem - 1);

                    /* For the current s', Q values for each action are stored in
                     * Q_xn. The optimal Bellman equation can be computed
                     * finding the maximum value inside Q_xn. They are zero if
                     * xn is an absorbing state.
                     */
                    outputs[miniBatchIndex](i) = miniBatchRewards(0, i) +
                                                 this->gamma * arma::as_scalar(QRegressorEnsemble.getRegressor(
                                                         1 - miniBatchIndex)(nextState, FiniteAction(maxIndex(index))));
                }
                else
                	outputs[miniBatchIndex](i) = miniBatchRewards(0, i);
            }

            BatchDataSimple featureDataset(miniBatchInput, outputs[miniBatchIndex]);
            QRegressorEnsemble.trainFeatures(featureDataset);
        }

        MiniBatchData::cleanMiniBatches(miniBatches);
    }

protected:
    DoubleFQIEnsemble QRegressorEnsemble;
    bool shuffle;
};

}

#endif /* INCLUDE_RELE_ALGORITHMS_BATCH_DOUBLEFQI_H_ */
