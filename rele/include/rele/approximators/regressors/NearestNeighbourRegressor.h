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

#include "Features.h"
#include <armadillo>
#include <vector>
#include <cassert>
#include "Regressors.h"

namespace ReLe
{

template<class InputC, bool denseOutput = true>
class NearestNeighbourRegressor_: public NonParametricRegressor_<InputC, denseOutput>
{

public:
    NearestNeighbourRegressor_(Features_<InputC, denseOutput>& phi, unsigned int k)
        : ParametricRegressor_<InputC>(phi.cols()), phi(phi), k(k)
    {
    }

    ~NearestNeighbourRegressor_()
    {
    }

    arma::vec operator()(const InputC& input)
    {
    	arma::vec features = phi(input);

    	unsigned int index = findNearestCluster(features, centroids);

    	return centroids.col(index);

    }

    void train(const std::vector<InputC>& samples)
    {
        unsigned int N = samples.size();

        //compute features matrix
        arma::mat features(phi.rows(), N);
        for(int i = 0; i < N; i++)
        {
            features.col(i) = phi(samples[i]);
        }

        arma::mat&& centroids = initRandom(features);

        //iteratively find better centroids
        bool hasConverged = false;

        arma::sp_mat oldClusters(N, k);

        do
        {
        	//create clusters
            arma::sp_umat&& clusters = createClusters(features, centroids);

            //for every cluster, re-calculate its centroid.
			recomputeCentroids(clusters, features, centroids);

            //check convergence
            hasConverged = arma::all(clusters == oldClusters);

            //save clusters
            oldClusters = clusters;
        }
        while(hasConverged);

        //save centroids
        this->centroids = centroids;
    }

    arma::sp_umat getClusters()
    {
    	return clusters;
    }

    arma::vec getCentroids()
    {
    	return centroids;
    }

private:
    arma::mat initRandom(const arma::mat& features)
    {
        arma::mat centroids;

        std::set<unsigned int> centroidsIndx;

        for (unsigned int i = 0; i < k; i++)
        {
            unsigned int indx;
            do
            {
                indx = RandomGenerator::sampleUniformInt(0, features.n_cols);
            }
            while (centroidsIndx.count(indx) == 0);

            centroidsIndx.insert(indx);
            centroids.col(i) = features.col(indx);
        }

        return centroids;
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

	void recomputeCentroids(const arma::sp_mat& clusters, const arma::mat& features,
				arma::mat& centroids)
	{
		//for every cluster, re-calculate its centroid.
		for (int i = 0; i < k; i++)
		{
			arma::uvec cluster_i = clusters.col(i);
			arma::mat clusterElements = features.cols(arma::find(cluster_i));
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

private:
    unsigned int k;
    Features_<InputC, denseOutput>& phi;
    arma::vec centroids;
    arma::sp_umat clusters;
};

typedef NearestNeighbourRegressor_<arma::vec> NearestNeighbourRegressor;

}


#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_NEARESTNEIGHBOURREGRESSOR_H_ */
