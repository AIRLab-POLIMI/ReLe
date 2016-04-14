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

#include "rele/algorithms/batch/td/W-FQI.h"

namespace ReLe
{
W_FQI::W_FQI(GaussianProcess& QRegressor,
             unsigned int nActions,
             double epsilon) :
    FQI(QRegressor, nActions, epsilon),
    nUpdatesQ(0)
{
    idxs = arma::mat(nActions, nActions - 1, arma::fill::zeros);
    arma::vec actions = arma::linspace(0, idxs.n_cols, idxs.n_rows);
    for(unsigned int i = 0; i < idxs.n_rows; i++)
        idxs.row(i) = actions(arma::find(actions != i)).t();
}

void W_FQI::step()
{
    arma::mat outputs(1, this->nSamples, arma::fill::zeros);

    for(unsigned int i = 0; i < this->nSamples; i++)
    {
        DenseState nextState = DenseState(this->nextStates(0, i));
        if(this->absorbingStates.count(i) == 0 && !this->firstStep)
        {
            arma::vec integrals(this->nActions, arma::fill::zeros);
            arma::vec means(this->nActions, arma::fill::zeros);
            arma::vec sigma(this->nActions, arma::fill::zeros);
            for(unsigned int j = 0; j < this->nActions; j++)
            {
                arma::vec results(2, arma::fill::zeros);
                results = this->QRegressor(nextState, FiniteAction(j));
                means(j) = results(0);
                sigma(j) = sqrt(results(1));
            }
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

    nUpdatesQ++;

    this->firstStep = false;

    this->checkCond();
}
}
