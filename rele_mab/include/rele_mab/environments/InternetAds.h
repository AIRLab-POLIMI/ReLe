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

#ifndef INCLUDE_RELE_ENVIRONMENTS_MAB_INTERNETADS_H_
#define INCLUDE_RELE_ENVIRONMENTS_MAB_INTERNETADS_H_

#include "rele_mab/environments/DiscreteMAB.h"


namespace ReLe
{

class InternetAds: public DiscreteMAB
{

public:
    enum ExperimentLabel
    {
        First, Second
    };

public:
    InternetAds(unsigned int nAds, ExperimentLabel experimentType = First);
    unsigned int getVisitors();

protected:
    unsigned int visitors;
};

}

#endif /* INCLUDE_RELE_ENVIRONMENTS_MAB_INTERNETADS_H_ */
