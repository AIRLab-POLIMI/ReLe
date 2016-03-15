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

#include "rele/feature_selection/PrincipalComponentAnalysis.h"


namespace ReLe
{

PrincipalComponentAnalysis::PrincipalComponentAnalysis(unsigned int k, bool useCorrelation)
    : k(k), useCorrelation(useCorrelation)
{

}

void PrincipalComponentAnalysis::createFeatures(const arma::mat& features)
{
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

    T = A.cols(0, k-1).t();

    arma::mat normalizedFeatures = features.each_col() - arma::mean(features, 1);


    newFeatures = T*normalizedFeatures;

}

arma::mat PrincipalComponentAnalysis::getTransformation()
{
    return T;
}

arma::mat PrincipalComponentAnalysis::getNewFeatures()
{
    return newFeatures;
}


}
