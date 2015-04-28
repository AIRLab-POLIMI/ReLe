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

//#define DEBUG_CSV_PARSER

namespace ReLe
{

bool CSVutils::readCSVLine(std::istream& is, vector<std::string>& tokens)
{
    std::string line;

    while(line.empty() && is)
    {
    	std::getline(is,line);
#ifdef DEBUG_CSV_PARSER
    	cout << "# " << line << std::endl;
#endif
    }

    if(line.empty())
    	return false;

    std::stringstream lineStream(line);
    std::string token;

    while(std::getline(lineStream, token, ','))
        tokens.push_back(token);

    return true;
}

void CSVutils::matrixToCSV(const mat& M, ostream& os)
{
    for (unsigned int i = 0; i < M.n_rows; i++)
    {
        int j, je = M.n_cols - 1;
        for (j = 0; j < je; j++)
        {
            os << M(i, j) << ",";
        }

        os << M(i, j) << endl;
    }
}

void CSVutils::vectorToCSV(const vec& v, ostream& os)
{
    int i, ie = v.n_elem - 1;
    for(i = 0; i < ie; i++)
    {
        os << v[i] << ",";
    }

    os << v[i] << endl;
}

void CSVutils::vectorToCSV(const std::vector<double> &v, ostream &os)
{
    int i, ie = v.size() - 1;
    for(i = 0; i < ie; i++)
    {
        os << v[i] << ",";
    }

    os << v[i] << endl;
}

}
