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

#ifndef INCLUDE_RELE_UTILS_CSV_H_
#define INCLUDE_RELE_UTILS_CSV_H_

#include <armadillo>
#include <iostream>

namespace ReLe
{

/*!
 * This class offers some helpful functions to manage CSV files.
 */
class CSVutils
{
public:
    /*!
     * Read a line from a CSV file.
     * \param is the CSV file to read
     * \param tokens vector of elements contained in the CSV file
     * \return a bool value indicating whether the read has been successful or not
     */
    static bool readCSVLine(std::istream& is, std::vector<std::string>& tokens);

    /*!
     * Print a CSV file from an armadillo matrix.
     * \param M matrix to print in the file
     * \param os the file where the matrix is printed
     */
    static void matrixToCSV(const arma::mat& M, std::ostream& os);

    /*!
     * Print a CSV file from an armadillo vector.
     * \param v vector to print in the file
     * \param os the file where the vector is printed
     */
    static void vectorToCSV(const arma::vec& v, std::ostream& os);

    /*!
     * Template function to print a CSV file from an armadillo matrix.
     * \param v vector to print in the file
     * \param os the file where the vector is printed
     */
    template<class T>
    static void vectorToCSV(const std::vector<T>& v, std::ostream& os)
    {
        int i, ie = v.size() - 1;
        for(i = 0; i < ie; i++)
        {
            os << v[i] << ",";
        }

        os << v[i] << std::endl;
    }
};

}


#endif /* INCLUDE_RELE_UTILS_CSV_H_ */
