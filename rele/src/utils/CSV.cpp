/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo
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

#include "CSV.h"

using namespace std;
using namespace arma;

namespace ReLe
{

void CSVutils::matrixToCSV(const mat& M, ostream& os)
{
    for (unsigned int i = 0; i < M.n_rows; i++)
    {
        unsigned int j;
        for (j = 0; j < M.n_cols - 1; j++)
        {
            os << M(i, j) << ", ";
        }

        os << M(i, j) << endl;
    }
}

void CSVutils::vectorToCSV(const vec& v, ostream& os)
{
    unsigned int i;
    for(i = 0; i < v.n_elem -1; i++)
    {
        os << v[i] << ", ";
    }

    os << v[i] << endl;
}

}