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

#include "MAB/InternetAds.h"


namespace ReLe
{

InternetAds::InternetAds(unsigned int nAds, double gamma, ExperimentLabel experimentType) :
    SimpleMAB(P, 1, gamma)
{
    EnvironmentSettings& task = getWritableSettings();
    task.finiteActionDim = nAds;
    if(experimentType == First)
    {
        P = P.ones(nAds) * 0.5;
        task.horizon = 100000;
    }
    else
    {
        P = arma::randu(nAds);
        P = 0.02 + (0.05 - 0.02) * P;
        task.horizon = 300000;
    }
}

}
