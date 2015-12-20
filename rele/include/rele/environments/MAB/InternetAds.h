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

#include "MAB/SimpleMAB.h"


namespace ReLe
{

class InternetAds: public SimpleMAB
{

    /*
     * This class is very related to the experiments presented in
     * "Estimating the Maximum Expected Value: An Analysis of (Nested) Cross
     * Validation and the Maximum Sample Average" (Hado Van Hasselt). Thus, it has not to be
     * used as a general interface for internet ads experiments. Nevertheless,
     * it can be easily changed for other type of experiments.
     */

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
