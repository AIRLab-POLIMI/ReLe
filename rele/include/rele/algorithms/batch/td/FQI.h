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

template<class StateC>
class FQI : public BatchAgent<FiniteAction, StateC>
{
public:

    FQI(BatchRegressor& QRegressor, unsigned int nStates, unsigned int nActions,
    		double gamma, unsigned int nMiniBatches = 1) :
    	BatchAgent<FiniteAction, StateC>(gamma, nMiniBatches),
		featureDatasetStart(nullptr),
        QRegressor(QRegressor),
        nStates(nStates),
        nActions(nActions)
    {
        QTable.zeros(nStates, nActions);
        QHat.zeros(this->nSamples);
    }

    virtual void init(Dataset<FiniteAction, StateC>& data) override
    {
        nSamples = 0;
        unsigned int nEpisodes = data.size();
        for(unsigned int k = 0; k < nEpisodes; k++)
            nSamples += data[k].size();

        unsigned int i = 0;
        nextStates.zeros(this->nSamples);
        for(auto& episode : data)
            for(auto& tr : episode)
            {
                if(tr.xn.isAbsorbing())
                    absorbingStates.insert(tr.xn);

                nextStates(i) = tr.xn;
                i++;
            }

        arma::mat rewards = data.rewardAsMatrix();
        arma::mat input = data.featuresAsMatrix(QRegressor.getFeatures());

    	if(featureDatasetStart)
    		delete featureDatasetStart;

        featureDatasetStart = new BatchDataSimple(input, rewards);
    }

    virtual void step() override
    {
        std::vector<arma::mat> outputs;
        std::vector<arma::vec> nextStatesMiniBatch;

        auto&& miniBatches = featureDatasetStart->getNMiniBatches(this->nMiniBatches);
        for(unsigned int i = 0; i < this->nMiniBatches; i++)
        {
            QRegressor.trainFeatures(*miniBatches[i]);
            nextStatesMiniBatch.push_back(nextStates(miniBatches[i]->getIndexes()));
            outputs.push_back(miniBatches[i]->getOutputs());
        }

        arma::mat input = miniBatches[0]->getFeatures();
        arma::mat rewards = miniBatches[0]->getOutputs();

        for(unsigned int i = 0; i < input.n_cols; i++)
        {
            arma::vec Q_xn(nActions, arma::fill::zeros);
            FiniteState nextState = FiniteState(nextStatesMiniBatch[0](i));
            if(absorbingStates.count(nextState) == 0)
                for(unsigned int u = 0; u < nActions; u++)
                    Q_xn(u) = arma::as_scalar(QRegressor(nextState,
                                                         FiniteAction(u)));

            outputs[0](i) = rewards(0, i) + this->gamma * arma::max(Q_xn);
        }

        BatchDataSimple featureDataset(input, outputs[0]);
        QRegressor.trainFeatures(featureDataset);

        double J;
        printInfo(J);

        MiniBatchData::cleanMiniBatches(miniBatches);
    }

    virtual void computeQHat()
    {
        unsigned int i = 0;
        /*for(auto& episode : this->data)
            for(auto& tr : episode)
            {
                QHat(i) = arma::as_scalar(QRegressor(tr.x, tr.u));
                i++;
            }*/
    }

    virtual void computeQTable()
    {
        for(unsigned int i = 0; i < nStates; i++)
            for(unsigned int j = 0; j < nActions; j++)
                QTable(i, j) = arma::as_scalar(QRegressor(FiniteState(i), FiniteAction(j)));
    }

    virtual void printInfo(double& J1)
    {
        arma::vec prevQHat = QHat;

        computeQHat();

        J1 = arma::norm(QHat - prevQHat);

        /*double J2 = 0;
        for(unsigned int i = 0; i < this->nMiniBatches; i++)
            J2 += arma::sum(
                      arma::square(QHat(this->miniBatches[i]->getIndexes()).t() - this->outputs[i]));
        J2 /= this->nSamples;

        arma::vec temp = QHat(this->miniBatches[0]->getIndexes());
        std::cout << std::endl << "Some Q-values (approximation, target):" << std::endl;
        for(unsigned int i = 0; i < 40; i++)
            std::cout << "(" << this->miniBatches[0]->getInput(i)(0) << ", " << this->miniBatches[0]->getInput(i)(1) <<
                      ", " << nextStatesMiniBatch[0](i) << ")  " << temp(i) << this->outputs[0].col(i) << std::endl;
        std::cout << "QHat - Previous QHat: " << J1 << std::endl;
        std::cout << "QHat - Q Bellman: " << J2 << std::endl;*/
    }

    virtual void printPolicy()
    {
        computeQTable();
        std::cout << std::endl << "Policy:" << std::endl;
        for(unsigned int i = 0; i < nStates; i++)
        {
            arma::uword policy;
            QTable.row(i).max(policy);
            std::cout << "policy(" << i << ") = " << policy << std::endl;
        }
    }

    arma::mat& getQ()
    {
        return QTable;
    }

    virtual ~FQI()
    {
    	if(featureDatasetStart)
    		delete featureDatasetStart;
    }

protected:
    arma::vec QHat;
    arma::mat QTable;
    arma::vec nextStates;
    BatchRegressor& QRegressor;
    unsigned int nSamples;
    unsigned int nStates;
    unsigned int nActions;
    std::set<FiniteState> absorbingStates;
    BatchDataSimple* featureDatasetStart;
};

}

#endif //INCLUDE_RELE_ALGORITHMS_BATCH_FQI_H_
