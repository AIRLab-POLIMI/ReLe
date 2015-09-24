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

#ifndef INCLUDE_RELE_IRL_FEATURE_SELECTION_PRINCIPALFEATUREANALYSIS_H_
#define INCLUDE_RELE_IRL_FEATURE_SELECTION_PRINCIPALFEATUREANALYSIS_H_

#include "regressors/NearestNeighbourRegressor.h"
#include "basis/IdentityBasis.h"
#include "features/DenseFeatures.h"

namespace ReLe
{

class PrincipalFeatureAnalysis
{
public:

    static arma::uvec selectFeatures(arma::mat& features, double varMin)
    {
        std::cout << "meanFeature" << std::endl << arma::sum(features,1)/ features.n_cols << std::endl;
        //compute covariance of features
        arma::mat Sigma = arma::cov(features.t());
        std::cout << "Sigma" << std::endl << Sigma << std::endl;

        //compute eigenvalues and eigenvectors
        arma::vec s;
        arma::mat A;

        arma::eig_sym(s, A, Sigma);
        s = arma::sort(s, "descend");
        A = fliplr(A);

        //compute minimum features dimensions
        unsigned int q = computeDimensions(s, varMin);

        //select features
        arma::mat Aq = A.cols(0, q-1).t();


        if(q > 1)
        {
            arma::mat means;
            arma::uvec clustersIndexes;
            cluster(Aq, means, clustersIndexes, q);


            arma::uvec idx = arma::linspace<arma::uvec>(0, features.n_rows -1, features.n_rows);

            arma::uvec selectedFeatures(q);

            for (unsigned int i = 0; i < q; i++)
            {
                arma::uvec elements = arma::find(clustersIndexes == i);
                arma::mat clusterVectors = Aq.cols(elements);
                arma::uvec vectorIdx = idx(elements);

                unsigned int k = findNearest(clusterVectors, means.col(i));
                selectedFeatures(i) = vectorIdx(k);
            }

            return arma::sort(selectedFeatures);
        }
        else
        {
            arma::vec mean = arma::sum(Aq, 1) / Aq.n_cols;

            arma::uvec index(1);

            index(0) = findNearest(Aq, mean);

            return index;
        }

    }

private:
    static unsigned int computeDimensions(arma::vec& s, double varMin)
    {
        std::cout << "s: " << s.t() << std::endl;
        unsigned int q;
        for (q = 0; q < s.n_elem; q++)
        {
            double var = arma::sum(s(arma::span(0, q)))/arma::sum(s);
            std::cout << "var: " << var << std::endl;
            if(var > varMin)
                break;
        }

        return q+1;
    }

    static void cluster(arma::mat& data, arma::mat& means, arma::uvec& clustersIndexes, unsigned int k)
    {
        BasisFunctions basis = IdentityBasis::generate(data.n_rows);
        DenseFeatures phi(basis);

        NearestNeighbourRegressor regressor(phi, k);
        regressor.setIterations(10);

        regressor.trainFeatures(data);

        means = regressor.getCentroids();
        clustersIndexes = regressor.getClustersIndexes();
    }

    static unsigned int findNearest(const arma::mat& elements, const arma::vec mean)
    {
        unsigned int minIndex = 0;
        double minDistance = std::numeric_limits<double>::infinity();

        for (unsigned int j = 0; j < elements.n_cols; j++)
        {
            arma::vec delta = mean - elements.col(j);
            double distance = arma::as_scalar(delta.t() * delta);
            if (distance < minDistance)
            {
                minDistance = distance;
                minIndex = j;
            }
        }

        return minIndex;
    }

};

}


#endif /* INCLUDE_RELE_IRL_FEATURE_SELECTION_PRINCIPALFEATUREANALYSIS_H_ */
