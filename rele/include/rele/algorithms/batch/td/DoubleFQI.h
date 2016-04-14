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
                      BatchRegressor& QRegressorB);

    virtual void writeOnStream(std::ofstream& out) override;
    virtual void readFromStream(std::ifstream& in) override;

    virtual ~DoubleFQIEnsemble();
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
class DoubleFQI: public FQI
{
public:
    /*!
     * Constructor.
     * \param QRegressorA the first regressor
     * \param QRegressorB the second regressor
     * \param nStates the number of states
     * \param nActions the number of actions
     * \param epsilon coefficient used to check whether to stop the training
     * \param shuffle if true, each regressor takes a different half of the dataset
     *        at each iteration
     */
    DoubleFQI(BatchRegressor& QRegressorA,
              BatchRegressor& QRegressorB,
              unsigned int nActions,
              double epsilon,
              bool shuffle = false);

    virtual void step() override;

protected:
    DoubleFQIEnsemble QRegressorEnsemble;
    std::vector<arma::uvec> indexes;
    bool shuffle;
};

}

#endif /* INCLUDE_RELE_ALGORITHMS_BATCH_DOUBLEFQI_H_ */
