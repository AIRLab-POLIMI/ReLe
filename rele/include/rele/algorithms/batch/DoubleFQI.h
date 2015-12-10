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

#include "FQI.h"
#include "Ensemble.h"


namespace ReLe
{

class DoubleFQIEnsemble : public Ensemble
{
public:
    DoubleFQIEnsemble(BatchRegressor& QRegressorA,
                      BatchRegressor& QRegressorB) :
        Ensemble(QRegressorA.getBasis(), 1)
    {
        regressors.push_back(&QRegressorA);
        regressors.push_back(&QRegressorB);
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
              double gamma) :
        FQI<StateC>(data, QRegressorEnsemble, nStates, nActions, gamma),
		QRegressorEnsemble(QRegressorA, QRegressorB)
    {
    }

    void step(arma::mat input, arma::mat& output, const arma::mat rewards) override
    {
        unsigned int selectedQ = RandomGenerator::sampleUniformInt(0, 1);

        if(selectedQ == 0)
        	doubleFQIStep(QRegressorEnsemble.getRegressor(0), QRegressorEnsemble.getRegressor(1), input, output, rewards);
        else
            doubleFQIStep(QRegressorEnsemble.getRegressor(1), QRegressorEnsemble.getRegressor(0), input, output, rewards);
    }

    void doubleFQIStep(BatchRegressor& trainingRegressor,
                       BatchRegressor& evaluationRegressor,
                       arma::mat input,
                       arma::mat& output,
                       const arma::mat rewards)
    {
        // Loop on each dataset sample (i.e. on each transition)
        unsigned int i = 0;

        for(auto& episode : this->data)
        {
            for(auto& tr : episode)
            {
                /* For the current s', Q values for each action are stored in
                 * Q_xn. The optimal Bellman equation can be computed
                 * finding the maximum value inside Q_xn. They are zero if
                 * xn is an absorbing state. Note that here we exchange the
                 * regressor according to Double Q-Learning algorithm.
                 */
                arma::vec Q_xn(this->nActions, arma::fill::zeros);
                if(!tr.xn.isAbsorbing())
                    for(unsigned int u = 0; u < this->nActions; u++)
                        Q_xn(u) = arma::as_scalar(trainingRegressor(tr.xn, FiniteAction(u)));

                double qmax = Q_xn.max();
                arma::uvec maxIndex = arma::find(Q_xn == qmax);
                unsigned int index = RandomGenerator::sampleUniformInt(0,
                                     maxIndex.n_elem - 1);
                output(i) = arma::as_scalar(rewards(0, i) + this->gamma * evaluationRegressor(tr.xn, FiniteAction(index)));

                i++;
            }
        }

        // The regressor is trained
        BatchDataFeatures<arma::vec, arma::vec> featureDataset(input, output);
        trainingRegressor.trainFeatures(featureDataset);
    }

protected:
    DoubleFQIEnsemble QRegressorEnsemble;
};

}

#endif /* INCLUDE_RELE_ALGORITHMS_BATCH_DOUBLEFQI_H_ */
