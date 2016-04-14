/*
*  rele,
*
*
*  Copyright (C) 2015 Davide Tateo & Matteo Pirotta
*  Versione 1.0
*
*  This file is part of rele.
*
*  rele is free software: you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation, either version 3 of the License, or
*  (at your option) any later version.
*
*  rele is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with rele.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef INCLUDE_RELE_ALGORITHMS_BATCH_FQI_H_
#define INCLUDE_RELE_ALGORITHMS_BATCH_FQI_H_

#include "rele/algorithms/batch/td/BatchTDAgent.h"

namespace ReLe
{

/*!
 * This class implements the output data for all Fitted Q-Iteration algorithms.
 * All version of Fitted Q-Iteration algorithms should use, or extend this class
 */
class FQIOutput : virtual public AgentOutputData
{
public:
    /*!
     * Constructor.
     * \param isFinal whether the data logged comes from the end of a run of the algorithm
     * \param gamma the discount factor
     * \param QRegressor the regressor
     */
    FQIOutput(bool isFinal, double gamma, BatchRegressor& QRegressor);

    virtual void writeData(std::ostream& os) override;
    virtual void writeDecoratedData(std::ostream& os) override;

protected:
    double gamma;
    BatchRegressor& QRegressor;
};

/*!
 * This class implements the Fitted Q-Iteration algorithm.
 * This algorithm is an off-policy batch algorithm that works with finite
 * action spaces.
 * It exploits the Bellman operator to build a dataset of Q-values from which
 * a regressor is trained.
 *
 * References
 * ==========
 *
 * [Ernst, Geurts, Wehenkel. Tree-Based Batch Mode Reinforcement Learning](http://www.jmlr.org/papers/volume6/ernst05a/ernst05a.pdf)
 */
class FQI : public BatchTDAgent<DenseState>
{
public:
    /*!
     * Constructor.
     * \param QRegressor the regressor
     * \param nStates the number of states
     * \param nActions the number of actions
     * \param epsilon coefficient used to check whether to stop the training
     */
    FQI(BatchRegressor& QRegressor, unsigned int nActions, double epsilon);

    virtual void init(Dataset<FiniteAction, DenseState>& data, EnvironmentSettings& envSettings) override;
    virtual void step() override;

    /*!
     * Check whether the stop condition is satisfied.
     */
    virtual void checkCond();

    /*!
     * Compute the Q-values approximation for each element in the dataset.
     */
    virtual void computeQHat();

    inline virtual AgentOutputData* getAgentOutputData() override
    {
        return new FQIOutput(false, this->gamma, this->QRegressor);
    }

    inline virtual AgentOutputData* getAgentOutputDataEnd() override
    {
        return new FQIOutput(true, this->gamma, this->QRegressor);
    }

protected:
    arma::vec QHat;
    arma::mat features;
    arma::mat states;
    arma::vec actions;
    arma::mat nextStates;
    arma::vec rewards;
    unsigned int nSamples;
    std::set<unsigned int> absorbingStates;
    double epsilon;
    bool firstStep;
};

}

#endif //INCLUDE_RELE_ALGORITHMS_BATCH_FQI_H_
