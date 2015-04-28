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

#ifndef INCLUDE_RELE_UTILS_CSV_H_
#define INCLUDE_RELE_UTILS_CSV_H_

#include <armadillo>
#include <iostream>

namespace ReLe
{

class CSVutils
{
public:
    static bool readCSVLine(std::istream& is, std::vector<std::string>& tokens);
    static void matrixToCSV(const arma::mat& M, std::ostream& os);
    static void vectorToCSV(const arma::vec& v, std::ostream& os);
    static void vectorToCSV(const std::vector<double>& v, std::ostream& os);

};

}


#endif /* INCLUDE_RELE_UTILS_CSV_H_ */
