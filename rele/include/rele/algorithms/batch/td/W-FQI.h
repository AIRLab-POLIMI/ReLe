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

#ifndef INCLUDE_RELE_ALGORITHMS_BATCH_W_FQI_H_
#define INCLUDE_RELE_ALGORITHMS_BATCH_W_FQI_H_

#include "rele/algorithms/batch/td/FQI.h"
#include "rele/approximators/regressors/others/GaussianProcess.h"
#include <boost/math/distributions/normal.hpp>

namespace ReLe
{

/*class FiniteW_FQI: public W_FQI<FiniteState>
{
public:
    FiniteW_FQI(BatchRegressor& QRegressor,
                unsigned int nStates,
                unsigned int nActions,
                double epsilon) :
        W_FQI<FiniteState>(QRegressor, nActions, epsilon),
        Q(arma::mat(nStates, nActions, arma::fill::zeros)),
        Q2(arma::mat(nStates, nActions, arma::fill::zeros)),
        sampleStdQ(arma::mat(nStates, nActions).fill(this->stdInfValue)),
        nStates(nStates)
    {
    }

    void step() override
    {
        arma::mat outputs(1, this->nSamples, arma::fill::zeros);

        for(unsigned int i = 0; i < this->nSamples; i++)
        {
            FiniteState nextState = FiniteState(this->nextStates(0, i));
            if(this->absorbingStates.count(i) == 0 && !this->firstStep)
            {
                arma::vec integrals(this->nActions, arma::fill::zeros);
                arma::vec means = Q.row(nextState).t();
                arma::vec sigma = sampleStdQ.row(nextState).t();
                for(unsigned int j = 0; j < integrals.n_elem; j++)
                {
                    double pdfMean = means(j);
                    double pdfSampleStd = sigma(j);
                    double lowerLimit = pdfMean - sigmaBound * pdfSampleStd;
                    double upperLimit = pdfMean + sigmaBound * pdfSampleStd;

                    arma::vec trapz = arma::linspace(lowerLimit, upperLimit, nTrapz + 1);
                    double diff = trapz(1) - trapz(0);

                    double result = 0;
                    for(unsigned int t = 0; t < trapz.n_elem - 1; t++)
                    {
                        arma::vec cdfs(this->idxs.n_cols, arma::fill::zeros);
                        for(unsigned int k = 0; k < cdfs.n_elem; k++)
                        {
                            boost::math::normal cdfNormal(means(this->idxs(j, k)), sigma(this->idxs(j, k)));
                            cdfs(k) = cdf(cdfNormal, trapz(t));
                        }
                        boost::math::normal pdfNormal(pdfMean, pdfSampleStd);
                        double t1 = pdf(pdfNormal, trapz(t)) * arma::prod(cdfs);

                        for(unsigned int k = 0; k < cdfs.n_elem; k++)
                        {
                            boost::math::normal cdfNormal(means(this->idxs(j, k)), sigma(this->idxs(j, k)));
                            cdfs(k) = cdf(cdfNormal, trapz(t + 1));
                        }
                        double t2 = pdf(pdfNormal, trapz(t + 1)) * arma::prod(cdfs);

                        result += (t1 + t2) * diff * 0.5;
                    }

                    integrals(j) = result;
                }

                double W = arma::dot(means, integrals);

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
    arma::mat Q;
    arma::mat Q2;
    arma::mat sampleStdQ;
    unsigned int nStates;

protected:
    inline void updateMeanAndSampleStdQ()
    {
        this->computeQ();
        this->nUpdatesQ++;
        Q2 = arma::square(Q);

        if(this->nUpdatesQ > 1)
        {
            arma::mat var = (Q2 - arma::square(Q)) / this->nUpdatesQ;

            arma::uvec zeroIndexes = arma::find(var < this->stdZeroValue * this->stdZeroValue);
            arma::uvec nonZeroIndexes = arma::find(var >= this->stdZeroValue * this->stdZeroValue);
            sampleStdQ(nonZeroIndexes) = arma::sqrt(var(nonZeroIndexes));
            sampleStdQ(zeroIndexes).fill(this->stdZeroValue);
        }
    }

    virtual void computeQ()
    {
        for(unsigned int i = 0; i < nStates; i++)
            for(unsigned int j = 0; j < nActions; j++)
                Q(i, j) = arma::as_scalar(QRegressor(FiniteState(i), FiniteAction(j)));
    }
};*/

/*!
 * This class implements a version of Fitted Q-iteration (FQI) that
 * exploits the Weighted Estimator, as done in Weighted Q-Learning.
 * This algorithm computes an estimate of the maximum action-value
 * approximating it as weighted sum of action-values approximated by the regressor
 * where the weights are the probabilities of the respective action-value to be the maximum.
 * Being a modified version of Fitted Q-Iteration, this algorithms
 * deals only with finite action spaces.
 */
class W_FQI: public FQI
{
public:
    static constexpr double stdZeroValue = 1e-5;
    static constexpr double stdInfValue = 1e10;
    static constexpr double nTrapz = 100;
    static constexpr double sigmaBound = 5;

public:
    W_FQI(GaussianProcess& QRegressor,
          unsigned int nActions,
          double epsilon);

    virtual void step() override;

protected:
    arma::mat idxs;
    unsigned int nUpdatesQ;
};

}

#endif /* INCLUDE_RELE_ALGORITHMS_BATCH_W_FQI_H_ */
