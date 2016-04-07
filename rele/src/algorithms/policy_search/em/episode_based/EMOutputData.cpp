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

#include "rele/algorithms/policy_search/em/episode_based/EMOutputData.h"

#include "rele/utils/CSV.h"

using namespace std;

namespace ReLe
{

EMOutputData::EMOutputData(unsigned int nbIndividual, unsigned int nbParams,
                           unsigned int nbEvals) :BlackBoxOutputData(nbIndividual, nbParams, nbEvals)
{

}

void EMOutputData::writeData(std::ostream& os)
{
    os << metaParams.n_elem << endl;
    CSVutils::vectorToCSV(metaParams, os);

    os << individuals[0].Pparams.n_elem << endl;
    os << individuals[0].Jvalues.n_elem << endl;
    os << individuals.size() << endl;

    for (auto& individual : individuals)
    {
        CSVutils::vectorToCSV(individual.Pparams, os);
        CSVutils::vectorToCSV(individual.Jvalues, os);
    }
}

void EMOutputData::writeDecoratedData(std::ostream& os)
{
    //TODO [MINOR] implement
    writeData(os);
}

}
