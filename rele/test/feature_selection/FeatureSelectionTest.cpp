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
#include "rele/feature_selection/PrincipalComponentAnalysis.h"

using namespace ReLe;

int main(int argc, char *argv[])
{
    arma::mat features =
    {
        { 0.1,  0.2,    0.3},
        { 0.02, 0.045,  0.027},
        {1e-6, 0.1,    1e-9},
        {1e-5, 1e-7,   5e-8}
    };

    PrincipalFeatureAnalysis pfa1(0.9);
    PrincipalFeatureAnalysis pfa2(0.9, false);

    PrincipalComponentAnalysis pca1(2);
    PrincipalComponentAnalysis pca2(2, false);

    pfa1.createFeatures(features);
    pfa2.createFeatures(features);

    pca1.createFeatures(features);
    pca2.createFeatures(features);

    std::cout << "Initial features: " << std::endl << features << std::endl;

    std::cout << "selected features indexes, pfa - correlation" << std::endl << pfa1.getIndexes();
    std::cout << "selected features indexes, pfa - no correlation" << std::endl << pfa2.getIndexes();

    std::cout << "T, pfa - correlation" << std::endl << pfa1.getTransformation();
    std::cout << "T, pfa - no correlation" << std::endl << pfa2.getTransformation();
    std::cout << "T, pca - correlation" << std::endl << pca1.getTransformation();
    std::cout << "T, pca - no correlation" << std::endl << pca2.getTransformation();

    std::cout << "phiNew, pfa - correlation" << std::endl << pfa1.getNewFeatures();
    std::cout << "phiNew, pfa - no correlation" << std::endl << pfa2.getNewFeatures();
    std::cout << "phiNew, pca - correlation" << std::endl << pca1.getNewFeatures();
    std::cout << "phiNew, pca - no correlation" << std::endl << pca2.getNewFeatures();


}
