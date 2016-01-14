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
 * Written by: Carlo D'Eramo
 */

#ifndef INCLUDE_RELE_ALGORITHMS_BATCH_W_FQI_H_
#define INCLUDE_RELE_ALGORITHMS_BATCH_W_FQI_H_

#include "rele/algorithms/batch/FQI.h"
#include <gsl/gsl_integration.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

#define STD_ZERO_VALUE 1E-100
#define STD_INF_VALUE 1E100


struct pars
{
    unsigned int xn;
    unsigned int u;
    arma::mat idxs;
    arma::mat meanQ;
    arma::mat sampleStdQ;
};

double fun(double x, void* params)
{
    pars p = *(pars*) params;
    const unsigned int xn = p.xn;
    const unsigned int u = p.u;
    const arma::mat& idxs = p.idxs;
    const arma::mat& meanQ = p.meanQ;
    const arma::mat& sampleStdQ = p.sampleStdQ;

    arma::vec cdf(idxs.n_cols, arma::fill::zeros);
    for(unsigned int i = 0; i < cdf.n_elem; i++)
        cdf(i) = gsl_cdf_gaussian_P(x - meanQ(xn, idxs(u, i)), sampleStdQ(xn, idxs(u, i)));
    double f = gsl_ran_gaussian_pdf(x - meanQ(xn, u), sampleStdQ(xn, u)) * arma::prod(cdf);

    return f;
}


namespace ReLe
{

template<class StateC>
class W_FQI: public FQI<StateC>
{
public:
    W_FQI(Dataset<FiniteAction, StateC>& data,
          BatchRegressor& QRegressor,
          unsigned int nStates,
          unsigned int nActions,
          double gamma) :
        FQI<StateC>(data, QRegressor, nStates, nActions, gamma, 1),
        meanQ(arma::mat(nStates, nActions, arma::fill::zeros)),
        // There are 0 elements at the beginning, variance is infinite
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

    void step(std::vector<MiniBatchData*>& miniBatches, std::vector<arma::mat>& outputs) override
    {
        updateMeanAndSampleStdQ();

        arma::mat input = miniBatches[0]->getFeatures();
        arma::mat rewards = miniBatches[0]->getOutputs();

        pars p;
        p.meanQ = meanQ;
        p.sampleStdQ = sampleStdQ;
        p.idxs = idxs;

        gsl_integration_workspace* w = gsl_integration_workspace_alloc(1000);
        for(unsigned int i = 0; i < input.n_cols; i++)
        {
            unsigned int nextState = this->nextStatesMiniBatch[0](i);
            if(this->absorbingStates.count(nextState) == 0)
            {
                arma::vec integrals(this->nActions, arma::fill::zeros);
                for(unsigned int j = 0; j < integrals.n_elem; j++)
                {
                    gsl_function f;
                    f.function = &fun;
                    p.xn = nextState;
                    p.u = j;
                    f.params = &p;

                    double result, error;
                    double pdfMean = meanQ(nextState, j);
                    double pdfSampleStd = sampleStdQ(nextState, j);
                    double lowerLimit = pdfMean - 8 * pdfSampleStd;
                    double upperLimit = pdfMean + 8 * pdfSampleStd;
                    gsl_integration_qag(&f, lowerLimit, upperLimit, 0, 1e-8, 1000, 6, w, &result, &error);

                    integrals(j) = result;
                }
                double W = arma::dot(integrals, this->QTable.row(nextState));

                outputs[0](i) = rewards(0, i) + this->gamma * W;
            }
            else
                outputs[0](i) = rewards(0, i);
        }
        gsl_integration_workspace_free(w);

        BatchDataSimple featureDataset(input, outputs[0]);
        this->QRegressor.trainFeatures(featureDataset);
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
