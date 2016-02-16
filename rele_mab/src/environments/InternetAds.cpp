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

#include "rele_mab/environments/InternetAds.h"


namespace ReLe
{

InternetAds::InternetAds(unsigned int nAds, ExperimentLabel experimentType) :
    DiscreteMAB(arma::ones(nAds), 1)
{
    if(experimentType == First)
    {
        P = arma::vec(nAds, arma::fill::ones) * 0.5;
        visitors = 100000;
    }
    else
    {
        P = 0.02 + (0.05 - 0.02) * arma::vec(nAds, arma::fill::randu);
        visitors = 300000;
    }

    EnvironmentSettings& task = getWritableSettings();
    task.gamma = 0;
}

unsigned int InternetAds::getVisitors()
{
    return visitors;
}

}
