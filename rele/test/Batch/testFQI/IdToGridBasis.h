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

#ifndef SRC_TEST_BATCH_FQI_IDTOGRIDBASIS_H_
#define SRC_TEST_BATCH_FQI_IDTOGRIDBASIS_H_

#include "BatchRegressorTraits.h"
#include <cmath>


namespace ReLe
{

template<class InputC>
class IdToGridBasis_ : public BasisFunction_<InputC>
{
public:
    IdToGridBasis_(unsigned int gridRows, unsigned int gridCols, unsigned int goalRow, unsigned int goalCol) :
        gridRows(gridRows), gridCols(gridCols), goalRow(goalRow), goalCol(goalCol)
    {
    }

    virtual ~IdToGridBasis_()
    {
    }

    virtual double operator()(const InputC& input) override
    {
        double manDistance;
        unsigned int inputRow = floor(input(2) / gridCols);
        unsigned int inputCol = fmod(input(2), gridCols);

        manDistance = abs(inputRow - goalRow) + abs(inputCol - goalCol);

        return manDistance;
    }

    virtual void writeOnStream(std::ostream& out) override
    {
    }

    virtual void readFromStream(std::istream& in) override
    {
    }


private:
    unsigned int gridRows;
    unsigned int gridCols;
    unsigned int goalRow;
    unsigned int goalCol;
};
typedef IdToGridBasis_<arma::vec> IdToGridBasis;

}


#endif /* SRC_TEST_BATCH_FQI_IDTOGRIDBASIS_H_ */
