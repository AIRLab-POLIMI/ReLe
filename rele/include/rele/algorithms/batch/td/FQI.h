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
/*
 * Fitted Q-Iteration (FQI)
 *
 * "Tree-Based Batch Mode Reinforcement Learning"
 * Damien Ernst, Pierre Geurts, Louis Wehenkel
 * Journal of Machine Learning Research, 6, 2006, pp. 503-556.
 */

class FQIOutput : virtual public AgentOutputData
{
public:
    FQIOutput(bool isFinal, double gamma, const arma::mat& Q) :
        AgentOutputData(isFinal),
        gamma(gamma),
        Q(Q)
    {
    }

    virtual void writeData(std::ostream& os) override
    {
        os << "- Parameters" << std::endl;
        os << "gamma: " << gamma << std::endl;

        os << "- Policy" << std::endl;
        for(unsigned int i = 0; i < Q.n_rows; i++)
        {
            arma::uword policy;
            Q.row(i).max(policy);
            os << "policy(" << i << ") = " << policy << std::endl;
        }
    }

    virtual void writeDecoratedData(std::ostream& os) override
    {
        os << "- Parameters" << std::endl;
        os << "gamma: " << gamma << std::endl;

        os << "- Action-value function" << std::endl;
        for(unsigned int i = 0; i < Q.n_rows; i++)
            for(unsigned int j = 0; j < Q.n_cols; j++)
            {
                os << "Q(" << i << ", " << j << ") = " << Q(i, j) << std::endl;
            }

        os << "- Policy" << std::endl;
        for(unsigned int i = 0; i < Q.n_rows; i++)
        {
            arma::uword policy;
            Q.row(i).max(policy);
            os << "policy(" << i << ") = " << policy << std::endl;
        }
    }

protected:
    double gamma;
    arma::mat Q;
};

template<class StateC>
class FQI : public BatchAgent<FiniteAction, StateC>
{
public:

    FQI(BatchRegressor& QRegressor, unsigned int nStates, unsigned int nActions,
        double epsilon) :
        QRegressor(QRegressor),
        nStates(nStates),
        nActions(nActions),
        QTable(arma::mat(nStates, nActions, arma::fill::zeros)),
        nSamples(0),
        firstStep(true),
        epsilon(epsilon)
    {
    }

    virtual void init(Dataset<FiniteAction, StateC>& data, double gamma) override
    {
        this->gamma = gamma;
        features = data.featuresAsMatrix(QRegressor.getFeatures());
        nSamples = features.n_cols;
        states = arma::vec(nSamples, arma::fill::zeros);
        actions = arma::vec(nSamples, arma::fill::zeros);
        nextStates = arma::vec(nSamples, arma::fill::zeros);
        rewards = arma::vec(nSamples, arma::fill::zeros);
        QHat = arma::vec(nSamples, arma::fill::zeros);

        unsigned int i = 0;
        for(auto& episode : data)
            for(auto& tr : episode)
            {
                if(tr.xn.isAbsorbing())
                    absorbingStates.insert(tr.xn);

                states(i) = tr.x;
                actions(i) = tr.u;
                nextStates(i) = tr.xn;
                rewards(i) = tr.r[0];
                i++;
            }
    }

    virtual void step() override
    {
        arma::mat outputs(1, nSamples, arma::fill::zeros);

        for(unsigned int i = 0; i < nSamples; i++)
        {
            arma::vec Q_xn(nActions, arma::fill::zeros);
            FiniteState nextState = FiniteState(nextStates(i));
            if(absorbingStates.count(nextState) == 0 && !firstStep)
                for(unsigned int u = 0; u < nActions; u++)
                    Q_xn(u) = arma::as_scalar(QRegressor(nextState,
                                                         FiniteAction(u)));

            outputs(i) = rewards(i) + this->gamma * arma::max(Q_xn);
        }

        BatchDataSimple featureDataset(features, outputs);
        QRegressor.trainFeatures(featureDataset);

        firstStep = false;

        checkCond();
    }

    virtual void checkCond()
    {
        arma::vec prevQHat = QHat;

        computeQHat();

        if(arma::norm(QHat - prevQHat) < epsilon)
            this->converged = true;
    }

    virtual void computeQHat()
    {
        for(unsigned int i = 0; i < states.n_elem; i++)
            QHat(i) = arma::as_scalar(QRegressor(FiniteState(states(i)), FiniteAction(actions(i))));
    }

    virtual void computeQTable()
    {
        for(unsigned int i = 0; i < nStates; i++)
            for(unsigned int j = 0; j < nActions; j++)
                QTable(i, j) = arma::as_scalar(QRegressor(FiniteState(i), FiniteAction(j)));
    }

    inline virtual AgentOutputData* getAgentOutputData() override
    {
        computeQTable();
        return new FQIOutput(false, this->gamma, QTable);
    }

    inline virtual AgentOutputData* getAgentOutputDataEnd() override
    {
        computeQTable();
        return new FQIOutput(true, this->gamma, QTable);
    }

    arma::mat& getQ()
    {
        computeQTable();

        return QTable;
    }

    virtual Policy<FiniteAction, StateC>* getPolicy() override
    {
        // FIXME
        return nullptr;
    }

    virtual ~FQI()
    {
    }

protected:
    arma::vec QHat;
    arma::mat QTable;
    arma::mat features;
    arma::vec states;
    arma::vec actions;
    arma::vec nextStates;
    arma::vec rewards;
    BatchRegressor& QRegressor;
    unsigned int nSamples;
    unsigned int nStates;
    unsigned int nActions;
    std::set<FiniteState> absorbingStates;
    double epsilon;
    bool firstStep;
};

}

#endif //INCLUDE_RELE_ALGORITHMS_BATCH_FQI_H_
