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

#ifndef INCLUDE_RELE_APPROXIMATORS_REGRESSORS_NEARESTNEIGHBOURREGRESSOR_H_
#define INCLUDE_RELE_APPROXIMATORS_REGRESSORS_NEARESTNEIGHBOURREGRESSOR_H_

#include "rele/approximators/Features.h"
#include "rele/approximators/Regressors.h"

#include <armadillo>
#include <vector>
#include <cassert>

namespace ReLe
{

template<class InputC, bool denseOutput = true>
class NearestNeighbourRegressor_: public UnsupervisedBatchRegressor_<InputC, arma::vec, denseOutput>
{
    USE_UNSUPERVISED_REGRESSOR_MEMBERS(InputC, arma::vec, denseOutput)
    DEFINE_FEATURES_TYPES(denseOutput)
public:
    NearestNeighbourRegressor_(Features_<InputC, denseOutput>& phi, unsigned int k)
        : UnsupervisedBatchRegressor_<InputC, arma::vec, denseOutput>(phi, phi.cols()), k(k), iterations(1),
          centroids(phi.rows(), k, arma::fill::randn), wcss(std::numeric_limits<double>::infinity())
    {
        assert(k >= 2);
    }

    arma::vec operator()(const InputC& input) override
    {
        arma::vec features = this->phi(input);

        unsigned int index = findNearestCluster(features, centroids);

        return centroids.col(index);
    }

    virtual void trainFeatures(const FeaturesCollection& features) override
    {
        wcss = std::numeric_limits<double>::infinity();

        for(unsigned int it = 0; it < iterations; it++)
        {
            runKMeansIteration(features);
        }
    }

    inline arma::sp_umat getClusters()
    {
        return clusters;
    }

    inline arma::mat getCentroids()
    {
        return centroids;
    }

    inline double getWCSS()
    {
        return wcss;
    }

    inline arma::uvec getClustersIndexes()
    {
        arma::uvec indexes(clusters.n_rows);

        for(unsigned int i = 0; i < clusters.n_rows; i++)
        {
            arma::sp_uvec feature_i = clusters.row(i).t();
            indexes(i) = *feature_i.row_indices;
        }

        return indexes;
    }

    inline void setIterations(unsigned int iterations)
    {
        assert(iterations >= 1);
        this->iterations = iterations;
    }

    inline void setK(unsigned int k)
    {
        assert(k >= 2);
        this->k = k;
    }

    virtual ~NearestNeighbourRegressor_()
    {

    }

private:
    arma::mat initRandom(const arma::mat& features)
    {
        arma::mat centroids(features.n_rows, k);

        std::set<unsigned int> centroidsIndx;

        for (unsigned int i = 0; i < k; i++)
        {
            unsigned int indx;
            do
            {
                indx = RandomGenerator::sampleUniformInt(0, features.n_cols - 1);
            }
            while (centroidsIndx.count(indx) != 0);

            centroidsIndx.insert(indx);
            centroids.col(i) = features.col(indx);
        }

        return centroids;
    }

    const arma::uvec getNonzeroIndices(arma::sp_uvec& sparseVec)
    {
        return arma::uvec(const_cast<arma::uword*>(sparseVec.row_indices), sparseVec.n_nonzero, false);
    }

    arma::sp_umat createClusters(const arma::mat& features, const arma::mat& centroids)
    {
        arma::sp_umat clusters(features.n_cols, centroids.n_cols);

        for(unsigned int i = 0; i < features.n_cols; i++)
        {
            unsigned int minIndex = findNearestCluster(features.col(i), centroids);
            clusters(i, minIndex) = 1;
        }

        return clusters;
    }

    void recomputeCentroids(const arma::sp_umat& clusters, const arma::mat& features,
                            arma::mat& centroids)
    {
        //for every cluster, re-calculate its centroid.
        for (int i = 0; i < k; i++)
        {
            arma::sp_uvec cluster_i = clusters.col(i);
            arma::mat clusterElements = features.cols(getNonzeroIndices(cluster_i));
            arma::vec centroid = arma::sum(clusterElements, 1)
                                 / clusterElements.n_cols;
            centroids.col(i) = centroid;
        }
    }

    unsigned int findNearestCluster(const arma::vec& currentFeature,
                                    const arma::mat& centroids)
    {
        unsigned int minIndex = 0;
        double minDistance = std::numeric_limits<double>::infinity();

        for (unsigned int j = 0; j < centroids.n_cols; j++)
        {
            arma::vec delta = currentFeature - centroids.col(j);
            double distance = arma::as_scalar(delta.t() * delta);
            if (distance < minDistance)
            {
                minDistance = distance;
                minIndex = j;
            }
        }

        return minIndex;
    }

    double computeWCSS(const arma::mat& features, const arma::sp_umat& clusters, const arma::mat centroids)
    {
        double wcss = 0;

        for (int i = 0; i < k; i++)
        {
            arma::sp_uvec cluster_i = clusters.col(i);
            arma::mat delta = features.cols(getNonzeroIndices(cluster_i));
            delta.each_col() -= centroids.col(i);

            wcss += arma::sum(arma::sum(arma::square(delta)));
        }


        return wcss;
    }

    //TODO [MINOR] support sparse vectors properly
    void runKMeansIteration(const FeaturesCollection& features)
    {
        arma::mat&& centroids = initRandom(features);

        //iteratively find better centroids
        bool hasConverged = false;

        arma::sp_umat oldClusters(features.n_cols, k);

        do
        {
            //create clusters
            arma::sp_umat&& clusters = createClusters(features, centroids);

            //for every cluster, re-calculate its centroid.
            recomputeCentroids(clusters, features, centroids);

            //check convergence
            arma::sp_umat delta = oldClusters - clusters;
            hasConverged = delta.n_nonzero == 0;

            //save clusters
            oldClusters = clusters;
        }
        while(!hasConverged);

        //compute within cluster sum of squared distances
        double wcss = computeWCSS(features, oldClusters, centroids);

        //save centroids, clusters and wcss if new minimun is found
        if(wcss < this->wcss)
        {
            this->centroids = centroids;
            this->clusters = oldClusters;
            this->wcss = wcss;
        }
    }

private:
    unsigned int k;
    unsigned int iterations;
    arma::mat centroids;
    arma::sp_umat clusters;
    double wcss;
};

typedef NearestNeighbourRegressor_<arma::vec> NearestNeighbourRegressor;

}


#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_NEARESTNEIGHBOURREGRESSOR_H_ */
