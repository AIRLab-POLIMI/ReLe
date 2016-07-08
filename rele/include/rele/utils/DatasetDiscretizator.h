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

#ifndef INCLUDE_RELE_UTILS_DATASETDISCRETIZATOR_H_
#define INCLUDE_RELE_UTILS_DATASETDISCRETIZATOR_H_

#include "rele/core/Transition.h"
#include "rele/utils/Range.h"
#include "rele/approximators/tiles/BasicTiles.h"

#include <vector>

namespace ReLe
{

class DatasetDiscretizator
{
public:
    DatasetDiscretizator(const Range& range, unsigned int actions);
    DatasetDiscretizator(const std::vector<Range>& ranges, const std::vector<unsigned int>& actionsPerDim);
    Dataset<FiniteAction, DenseState> discretize(const Dataset<DenseAction, DenseState>& dataset);
    ~DatasetDiscretizator();

private:
    FiniteAction discretizeAction(const DenseAction& action);

private:
    BasicTiles tile;

};



}

#endif /* INCLUDE_RELE_UTILS_DATASETDISCRETIZATOR_H_ */
