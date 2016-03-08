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
 *
 * TODO: This version only works for tabular regression.
 * 		 Fix Q estimation as done in WQ-Learning.
 */

#ifndef INCLUDE_RELE_ALGORITHMS_BATCH_W_FQI_H_
#define INCLUDE_RELE_ALGORITHMS_BATCH_W_FQI_H_

#include "rele/algorithms/batch/td/FQI.h"
#include <boost/math/distributions/normal.hpp>

#define STD_ZERO_VALUE 1E-5
#define STD_INF_VALUE 1E10


namespace ReLe
{

template<class StateC>
class W_FQI: public FQI<StateC>
{
public:
    W_FQI(BatchRegressor& QRegressor,
          unsigned int nStates,
          unsigned int nActions,
          double epsilon) :
        FQI<StateC>(QRegressor, nStates, nActions, epsilon),
        meanQ(arma::mat(nStates, nActions, arma::fill::zeros)),
        sampleStdQ(arma::mat(nStates, nActions).fill(STD_INF_VALUE)),
        sumQ(arma::mat(nStates, nActions, arma::fill::zeros)),
        sumSquareQ(arma::mat(nStates, nActions, arma::fill::zeros)),
        nUpdatesQ(0)
    {
        idxs = arma::mat(nActions, nActions - 1, arma::fill::zeros);
        arma::vec actions = arma::linspace(0, idxs.n_cols, idxs.n_rows);
        for(unsigned int i = 0; i < idxs.n_rows; i++)
            idxs.row(i) = actions(arma::find(actions != i)).t();
    }

    void step() override
    {
        arma::mat outputs(1, this->nSamples, arma::fill::zeros);

        unsigned int nTrapz = 100;

        for(unsigned int i = 0; i < this->nSamples; i++)
        {
            FiniteState nextState = FiniteState(this->nextStates(i));
            if(this->absorbingStates.count(nextState) == 0 && !this->firstStep)
            {
                arma::vec integrals(this->nActions, arma::fill::zeros);
                for(unsigned int j = 0; j < integrals.n_elem; j++)
                {
            		arma::vec means = meanQ.row(nextState).t();
            		arma::vec sigma = sampleStdQ.row(nextState).t();
                    double pdfMean = means(j);
                    double pdfSampleStd = sigma(j);
                    double lowerLimit = pdfMean - 5 * pdfSampleStd;
                    double upperLimit = pdfMean + 5 * pdfSampleStd;

            		arma::vec trapz = arma::linspace(lowerLimit, upperLimit, nTrapz + 1);
            		double diff = trapz(1) - trapz(0);

            		double result = 0;
            		for(unsigned int t = 0; t < trapz.n_elem - 1; t++)
            		{
            			arma::vec cdfs(idxs.n_cols, arma::fill::zeros);
            			for(unsigned int k = 0; k < cdfs.n_elem; k++)
            			{
            				boost::math::normal cdfNormal(means(idxs(j, k)), sigma(idxs(j, k)));
            				cdfs(k) = cdf(cdfNormal, trapz(t));
            			}
            			boost::math::normal pdfNormal(pdfMean, pdfSampleStd);
            			double t1 = pdf(pdfNormal, trapz(t)) * arma::prod(cdfs);

            			for(unsigned int k = 0; k < cdfs.n_elem; k++)
            			{
            				boost::math::normal cdfNormal(means(idxs(j, k)), sigma(idxs(j, k)));
            				cdfs(k) = cdf(cdfNormal, trapz(t + 1));
            			}
            			double t2 = pdf(pdfNormal, trapz(t + 1)) * arma::prod(cdfs);

            			result += (t1 + t2) * diff * 0.5;
            		}

                    integrals(j) = result;
                }

                double W = arma::dot(this->QTable.row(nextState), integrals);

                outputs(i) = this->rewards(i) + this->gamma * W;
            }
            else
                outputs(i) = this->rewards(i);
        }

        BatchDataSimple featureDataset(this->features, outputs);
        this->QRegressor.trainFeatures(featureDataset);

        updateMeanAndSampleStdQ();

        this->firstStep = false;

        this->checkCond();
    }

protected:
    arma::mat idxs;
    arma::mat meanQ;
    arma::mat sampleStdQ;
    arma::mat sumQ;
    arma::mat sumSquareQ;
    unsigned int nUpdatesQ;

protected:
    inline void updateMeanAndSampleStdQ()
    {
        this->computeQTable();
        nUpdatesQ++;

        if(nUpdatesQ > 1)
        {
            sumQ += this->QTable;;
            sumSquareQ += arma::square(this->QTable);
            meanQ = sumQ / nUpdatesQ;
            arma::mat diff = sumSquareQ - arma::square(sumQ) / nUpdatesQ;

            sampleStdQ.fill(STD_ZERO_VALUE);
            arma::uvec indexes = arma::find(diff > 0);
            sampleStdQ(indexes) = sqrt((diff(indexes) / (nUpdatesQ - 1)) / nUpdatesQ);
        }
    }
};

}

#endif /* INCLUDE_RELE_ALGORITHMS_BATCH_W_FQI_H_ */
