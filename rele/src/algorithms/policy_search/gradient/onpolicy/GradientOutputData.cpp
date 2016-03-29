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

#include "rele/algorithms/policy_search/gradient/GradientOutputData.h"
#include "rele/utils/CSV.h"

using namespace std;
using namespace arma;

namespace ReLe
{


GradientIndividual::GradientIndividual()
    : AgentOutputData(true)
{

}

void GradientIndividual::writeData(ostream &os)
{
    os << estimated_gradient.n_elem << endl;
    int dim = history_J.size();
    os << dim << endl;
    CSVutils::vectorToCSV(history_J, os);
    for (int i = 0; i < dim; ++i)
    {
        CSVutils::vectorToCSV(history_gradients[i], os);
    }
    CSVutils::vectorToCSV(policy_parameters, os);
    CSVutils::vectorToCSV(estimated_gradient, os);
}

void GradientIndividual::writeDecoratedData(ostream &os)
{
    this->writeData(os);
}

}
