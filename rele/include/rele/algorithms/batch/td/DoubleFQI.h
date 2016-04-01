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

#ifndef INCLUDE_RELE_ALGORITHMS_BATCH_DOUBLEFQI_H_
#define INCLUDE_RELE_ALGORITHMS_BATCH_DOUBLEFQI_H_

#include "rele/algorithms/batch/td/FQI.h"
#include "rele/approximators/regressors/Ensemble.h"


namespace ReLe
{

/*!
 * This class implements an ensemble of regressors to be used
 * for the modified version of Fitted Q-Iteration algorithm
 * that uses the Double Estimator.
 */
class DoubleFQIEnsemble : public Ensemble
{
public:
	/*!
	 * Constructor.
	 * \param QRegressorA the first regressor of the ensemble
	 * \param QRegressorB the second regressor of the ensemble
	 */
    DoubleFQIEnsemble(BatchRegressor& QRegressorA,
                      BatchRegressor& QRegressorB) :
        Ensemble(QRegressorA.getFeatures())
    {
        regressors.push_back(&QRegressorA);
        regressors.push_back(&QRegressorB);
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
};

/*!
 * This class implements a version of Fitted Q-iteration (FQI) that
 * exploits the Double Estimator, as done in Double Q-Learning, using
 * two regressors. One of the regressor is used to select the action
 * with the highest action-value for the first regressor and the other
 * is used to compute the value of the selected actions.
 * Being a modified version of Fitted Q-Iteration, this algorithms
 * deals only with finite action spaces.
 */
template<class StateC>
class DoubleFQI: public FQI<StateC>
{
public:
	/*!
	 * Constructor.
	 * \param QRegressorA the first regressor
	 * \param QRegressorB the second regressor
	 * \param nStates the number of states
	 * \param nActions the number of actions
	 * \param epsilon coefficient used to check whether to stop the training
	 * \param shuffle if true, each regressor take a different half of the dataset
	 *        at each iteration
	 */
    DoubleFQI(BatchRegressor& QRegressorA,
              BatchRegressor& QRegressorB,
              unsigned int nStates,
              unsigned int nActions,
              double epsilon,
              bool shuffle = false) :
        FQI<StateC>(QRegressorEnsemble, nStates, nActions, epsilon),
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

        indexes.push_back(allIndexes(arma::span(0, floor(this->nSamples / 2) - 1)));
        indexes.push_back(allIndexes(arma::span(floor(this->nSamples / 2), this->nSamples - 1)));
        for(unsigned int i = 0; i < 2; i++)
        {
            arma::mat features = this->features.cols(indexes[i]);
            arma::vec rewards = this->rewards(indexes[i]);
            arma::vec nextStates = this->nextStates(indexes[i]);
            arma::mat outputs(1, indexes[i].n_elem, arma::fill::zeros);

            for(unsigned int j = 0; j < indexes[i].n_elem; j++)
            {
                FiniteState nextState = FiniteState(nextStates(j));
                if(this->absorbingStates.count(nextState) == 0 && !this->firstStep)
                {
                    arma::vec Q_xn(this->nActions, arma::fill::zeros);
                    for(unsigned int u = 0; u < this->nActions; u++)
                        Q_xn(u) = arma::as_scalar(
                                      QRegressorEnsemble.getRegressor(i)(
                                          nextState, FiniteAction(u)));

                    double qmax = Q_xn.max();
                    arma::uvec maxIndex = find(Q_xn == qmax);
                    unsigned int index = RandomGenerator::sampleUniformInt(0,
                                         maxIndex.n_elem - 1);

                    outputs(j) = rewards(j) + this->gamma * arma::as_scalar(
                                     QRegressorEnsemble.getRegressor(1 - i)(
                                         nextState, FiniteAction(maxIndex(index))));
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

protected:
    DoubleFQIEnsemble QRegressorEnsemble;
    std::vector<arma::uvec> indexes;
    bool shuffle;
};

}

#endif /* INCLUDE_RELE_ALGORITHMS_BATCH_DOUBLEFQI_H_ */
