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

#include "rele/IRL/feature_selection/PrincipalFeatureAnalysis.h"
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

    arma::uvec selectedFeatures1 = PrincipalFeatureAnalysis::selectFeatures(features, 0.9);
    std::cout << "selected features indexes" << std::endl << selectedFeatures1;

    arma::uvec selectedFeatures2 = PrincipalFeatureAnalysis::selectFeatures(features, 0.9, false);
    std::cout << "selected features indexes" << std::endl << selectedFeatures2;
}
