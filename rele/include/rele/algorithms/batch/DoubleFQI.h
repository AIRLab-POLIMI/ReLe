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

#include "FQI.h"
#include "Ensemble.h"


namespace ReLe
{

class DoubleFQIEnsemble : public Ensemble
{
public:
    DoubleFQIEnsemble(BatchRegressor& QRegressorA,
                      BatchRegressor& QRegressorB,
                      double k = 0.5) :
        Ensemble(QRegressorA.getFeatures()),
        k(k)
    {
        assert(k > 0 && k < 1);

        regressors.push_back(&QRegressorA);
        regressors.push_back(&QRegressorB);
    }

    void trainFeatures(BatchDataSimple& featureDataset) override
    {
        /* This function extract the respective dataset for each
         * regressor in the ensemble and train them.
         */
        arma::mat input = featureDataset.getFeatures();
        arma::mat output = featureDataset.getOutputs();

        unsigned int nSamples = input.n_cols;

        unsigned int splitIndex = nSamples * k;
        arma::mat inputA = input.cols(arma::span(0, splitIndex - 1));
        arma::mat inputB = input.cols(arma::span(splitIndex, nSamples - 1));
        arma::mat outputA = output.cols(arma::span(0, splitIndex - 1));
        arma::mat outputB = output.cols(arma::span(splitIndex, nSamples - 1));

        BatchDataSimple featureDatasetA(inputA, outputA);
        BatchDataSimple featureDatasetB(inputB, outputB);

        regressors[0]->trainFeatures(featureDatasetA);
        regressors[1]->trainFeatures(featureDatasetB);
    }

    double getK() const
    {
        return k;
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
    double k;
};

template<class StateC>
class DoubleFQI: public FQI<StateC>
{
public:

    /* This class implements the Double FQI algorithm. As a batch algorithm, it takes
     * a dataset of (s, a, r, s') transitions, together with the regressors that
     * it is used to approximate the target distribution of Q values.
     */
    DoubleFQI(Dataset<FiniteAction, StateC>& data,
              BatchRegressor& QRegressorA,
              BatchRegressor& QRegressorB,
              unsigned int nStates,
              unsigned int nActions,
              double gamma,
              bool shuffle = false) :
        FQI<StateC>(data, QRegressorEnsemble, nStates, nActions, gamma),
        QRegressorEnsemble(QRegressorA, QRegressorB),
        shuffle(shuffle)
    {
    }

    void step(arma::mat input, arma::mat& output, arma::mat rewards) override
    {
        double k = QRegressorEnsemble.getK();

        /* Dataset is shuffled if flag true.
         *
         * TODO: this way of shuffling is temporary. It will be
         * adapted to the changes that will be made in BatchData classes.
         */
        if(shuffle)
        {
            this->indexes = arma::shuffle(this->indexes);
            input = input.cols(this->indexes);
            rewards = rewards.cols(this->indexes);
            this->nextStates = this->nextStates(this->indexes);
        }

        // First regressor output creation
        for(unsigned int i = 0; i < this->nSamples * k; i++)
        {
            /* In order to be able to fill the output vector (i.e. regressor
             * target values), we need to compute the Q-values for each
             * s' sample in the dataset and for each action in the
             * set of actions of the problem. Recalling the fact that the values
             * are zero for each action in an absorbing state, we check if s' is
             * absorbing and, if it is, we leave the Q-values fixed to zero.
             */
            arma::vec Q_xn(this->nActions, arma::fill::zeros);
            if(!FiniteState(this->nextStates(i)).isAbsorbing())
                for(unsigned int u = 0; u < this->nActions; u++)
                    Q_xn(u) = arma::as_scalar(
                                  QRegressorEnsemble.getRegressor(0)(FiniteState(this->nextStates(i)),
                                          FiniteAction(u)));

            // Compute index of action with max Q-value in the next state
            double qmax = Q_xn.max();
            arma::uvec maxIndex = find(Q_xn == qmax);
            unsigned int index = RandomGenerator::sampleUniformInt(0,
                                 maxIndex.n_elem - 1);

            /* For the current s', Q values for each action are stored in
             * Q_xn. The optimal Bellman equation can be computed
             * finding the maximum value inside Q_xn. They are zero if
             * xn is an absorbing state.
             */
            output(i) = rewards(0, i) + this->gamma * arma::as_scalar(
                            QRegressorEnsemble.getRegressor(1)(FiniteState(this->nextStates(i)),
                                    FiniteAction(index)));
        }

        // Second regressor output creation
        for(unsigned int i = this->nSamples * k; i < this->nSamples; i++)
        {
            /* In order to be able to fill the output vector (i.e. regressor
             * target values), we need to compute the Q values for each
             * s' sample in the dataset and for each action in the
             * set of actions of the problem. Recalling the fact that the values
             * are zero for each action in an absorbing state, we check if s' is
             * absorbing and, if it is, we leave the Q-values fixed to zero.
             */
            arma::vec Q_xn(this->nActions, arma::fill::zeros);
            if(!FiniteState(this->nextStates(i)).isAbsorbing())
                for(unsigned int u = 0; u < this->nActions; u++)
                    Q_xn(u) = arma::as_scalar(
                                  QRegressorEnsemble.getRegressor(1)(FiniteState(this->nextStates(i)),
                                          FiniteAction(u)));

            double qmax = Q_xn.max();
            arma::uvec maxIndex = find(Q_xn == qmax);
            unsigned int index = RandomGenerator::sampleUniformInt(0,
                                 maxIndex.n_elem - 1);

            /* For the current s', Q values for each action are stored in
             * Q_xn. The optimal Bellman equation can be computed
             * finding the maximum value inside Q_xn. They are zero if
             * xn is an absorbing state.
             */
            output(i) = rewards(0, i) + this->gamma * arma::as_scalar(
                            QRegressorEnsemble.getRegressor(0)(FiniteState(this->nextStates(i)),
                                    FiniteAction(index)));
        }

        // The regressors are trained
        BatchDataSimple featureDataset(input, output);
        QRegressorEnsemble.trainFeatures(featureDataset);
    }

protected:
    DoubleFQIEnsemble QRegressorEnsemble;
    bool shuffle;
};

}

#endif /* INCLUDE_RELE_ALGORITHMS_BATCH_DOUBLEFQI_H_ */
