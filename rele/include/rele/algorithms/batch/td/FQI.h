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

#include "rele/core/BatchAgent.h"

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
    FQIOutput(bool isFinal, double gamma, BatchRegressor& QRegressor) :
        AgentOutputData(isFinal),
        gamma(gamma),
        QRegressor(QRegressor)
    {
    }

    virtual void writeData(std::ostream& os) override
    {
        os << "- Parameters" << std::endl;
        os << "gamma: " << gamma << std::endl;
    }

    virtual void writeDecoratedData(std::ostream& os) override
    {
        os << "- Parameters" << std::endl;
        os << "gamma: " << gamma << std::endl;
    }

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
template<class StateC>
class FQI : public BatchAgent<FiniteAction, StateC>
{
public:
    /*!
     * Constructor.
     * \param QRegressor the regressor
     * \param nStates the number of states
     * \param nActions the number of actions
     * \param epsilon coefficient used to check whether to stop the training
     */
    FQI(BatchRegressor& QRegressor, unsigned int nActions,
        double epsilon) :
        QRegressor(QRegressor),
        nActions(nActions),
        nSamples(0),
        firstStep(true),
        epsilon(epsilon)
    {
    }

    virtual void init(Dataset<FiniteAction, StateC>& data, EnvironmentSettings& envSettings) override
    {
        this->gamma = envSettings.gamma;
        features = data.featuresAsMatrix(QRegressor.getFeatures());
        nSamples = features.n_cols;
        states = arma::mat(envSettings.stateDimensionality,
                           nSamples,
                           arma::fill::zeros);
        actions = arma::vec(nSamples, arma::fill::zeros);
        nextStates = arma::mat(envSettings.stateDimensionality,
                               nSamples,
                               arma::fill::zeros);
        rewards = arma::vec(nSamples, arma::fill::zeros);
        QHat = arma::vec(nSamples, arma::fill::zeros);

        unsigned int i = 0;
        for(auto& episode : data)
            for(auto& tr : episode)
            {
                if(tr.xn.isAbsorbing())
                    absorbingStates.insert(i);

                states.col(i) = vectorize(tr.x);
                actions(i) = tr.u;
                nextStates.col(i) = vectorize(tr.xn);
                rewards(i) = tr.r[0];
                i++;
            }
    }

    virtual void step() override
    {
    	arma::vec outputs(nSamples, arma::fill::zeros);

        for(unsigned int i = 0; i < nSamples; i++)
        {
            arma::vec Q_xn(nActions, arma::fill::zeros);
            if(absorbingStates.count(i) == 0 && !firstStep)
            {
                for(unsigned int u = 0; u < nActions; u++)
                    Q_xn(u) = arma::as_scalar(QRegressor(nextStates.col(i),
                                                         FiniteAction(u)));

                outputs(i) = rewards(i) + this->gamma * arma::max(Q_xn);
            }
            else
                outputs(i) = rewards(i);
        }

        BatchDataSimple featureDataset(features, outputs);
        QRegressor.trainFeatures(featureDataset);

        firstStep = false;

        checkCond();
    }

    /*!
     * Check whether the stop condition is satisfied.
     */
    virtual void checkCond()
    {
        arma::vec prevQHat = QHat;

        computeQHat();

        if(arma::norm(QHat - prevQHat) < epsilon)
            this->converged = true;
    }

    /*!
     * Compute the Q-values approximation for each element in the dataset.
     */
    virtual void computeQHat()
    {
        for(unsigned int i = 0; i < states.n_elem; i++)
            QHat(i) = arma::as_scalar(QRegressor(states.col(i), FiniteAction(actions(i))));
    }

    inline virtual AgentOutputData* getAgentOutputData() override
    {
        return new FQIOutput(false, this->gamma, QRegressor);
    }

    inline virtual AgentOutputData* getAgentOutputDataEnd() override
    {
        return new FQIOutput(true, this->gamma, QRegressor);
    }

    virtual Policy<FiniteAction, StateC>* getPolicy() override
    {
        //TODO [INTERFACE] fix interface implementation for batch methods...
        return nullptr;
    }

    virtual ~FQI()
    {
    }

protected:
    arma::vec QHat;
    arma::mat features;
    arma::mat states;
    arma::vec actions;
    arma::mat nextStates;
    arma::vec rewards;
    BatchRegressor& QRegressor;
    unsigned int nSamples;
    unsigned int nActions;
    std::set<unsigned int> absorbingStates;
    double epsilon;
    bool firstStep;
};

}

#endif //INCLUDE_RELE_ALGORITHMS_BATCH_FQI_H_
