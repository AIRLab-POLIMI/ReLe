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

#include "rele/utils/DatasetDiscretizator.h"
#include "rele/approximators/Tiles.h"

namespace ReLe
{

DatasetDiscretizator::DatasetDiscretizator(const Range& range, unsigned int actions)
    : tile(range, actions)
{

}

DatasetDiscretizator::DatasetDiscretizator(const std::vector<Range>& ranges,
        const std::vector<unsigned int>& actionsPerDim)
    : tile(ranges, actionsPerDim)
{

}

Dataset<FiniteAction, DenseState> DatasetDiscretizator::discretize(const Dataset<DenseAction, DenseState>& dataset)
{
    Dataset<FiniteAction, DenseState> discreteDataset;
    discreteDataset.resize(dataset.size());

    for(unsigned int ep = 0; ep < dataset.size(); ep++)
    {
        auto& episode = dataset[ep];
        auto& discreteEpisode = discreteDataset[ep];

        discreteEpisode.resize(episode.size());

        for(unsigned int t = 0; t < episode.size(); t++)
        {
            auto& tr = episode[t];
            auto& discreteTr = discreteEpisode[t];

            discreteTr.x = tr.x;
            discreteTr.u = discretizeAction(tr.u);
            discreteTr.xn = tr.xn;
            discreteTr.r = tr.r;
        }
    }

    return discreteDataset;
}

DatasetDiscretizator::~DatasetDiscretizator()
{

}

FiniteAction DatasetDiscretizator::discretizeAction(const DenseAction& action)
{
    unsigned int actionNumber = tile(action);
    return FiniteAction(actionNumber);
}

}
