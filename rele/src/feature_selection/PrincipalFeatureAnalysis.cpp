/*
 * rele,
 *
 *
 * Copyright (C) 2016 Davide Tateo
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

#include "rele/feature_selection/PrincipalFeatureAnalysis.h"

#include "rele/approximators/regressors/others/NearestNeighbourRegressor.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/features/DenseFeatures.h"


namespace ReLe
{

PrincipalFeatureAnalysis::PrincipalFeatureAnalysis(double varMin, bool useCorrelation)
    : varMin(varMin), useCorrelation(useCorrelation)
{
    initialSize = 0;
}

void PrincipalFeatureAnalysis::createFeatures(const arma::mat& features)
{
    initialSize = features.n_rows;

    arma::mat Sigma;

    if(useCorrelation)
    {
        //compute correlation of features
        Sigma = arma::cor(features.t());
        Sigma = (Sigma + Sigma.t())/2;
    }
    else
    {
        //compute covariance of features
        Sigma = arma::cov(features.t());
    }

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

        indexes = arma::sort(selectedFeatures);
    }
    else
    {
        arma::vec mean = arma::mean(Aq, 1);

        arma::uvec index(1);

        index(0) = findNearest(Aq, mean);

        indexes = index;
    }

    newFeatures = features.rows(indexes);

}


arma::mat PrincipalFeatureAnalysis::getTransformation()
{
    arma::mat T(indexes.n_elem, initialSize, arma::fill::zeros);

    for(unsigned int i = 0; i < indexes.n_elem; i++)
    {
        auto indx = indexes(i);
        T(i, indx) = 1.0;
    }


    return T;
}

arma::mat PrincipalFeatureAnalysis::getNewFeatures()
{
    return newFeatures;
}

unsigned int PrincipalFeatureAnalysis::computeDimensions(arma::vec& s, double varMin)
{
    unsigned int q;
    for (q = 0; q < s.n_elem; q++)
    {
        double var = arma::sum(s(arma::span(0, q)))/arma::sum(s);
        if(var > varMin)
            break;
    }

    return q+1;
}

void PrincipalFeatureAnalysis::cluster(arma::mat& data, arma::mat& means, arma::uvec& clustersIndexes, unsigned int k)
{
    BasisFunctions basis = IdentityBasis::generate(data.n_rows);
    DenseFeatures phi(basis);

    NearestNeighbourRegressor regressor(phi, k);
    regressor.setIterations(10);

    regressor.trainFeatures(data);

    means = regressor.getCentroids();
    clustersIndexes = regressor.getClustersIndexes();
}

unsigned int PrincipalFeatureAnalysis::findNearest(const arma::mat& elements, const arma::vec mean)
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


}


