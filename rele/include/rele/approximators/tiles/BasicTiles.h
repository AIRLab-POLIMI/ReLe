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

#ifndef INCLUDE_RELE_APPROXIMATORS_TILES_BASICTILES_H_
#define INCLUDE_RELE_APPROXIMATORS_TILES_BASICTILES_H_

#include "Tiles.h"
#include "Range.h"

namespace ReLe
{

class BasicTiles : public Tiles
{
public:
    BasicTiles(Range& range, unsigned int tilesN);
    BasicTiles(std::vector<Range>& ranges, std::vector<unsigned int>& tilesN);
    inline virtual unsigned int size()
    {
        return tilesSize;
    }
    virtual unsigned int operator()(const arma::vec& input);
    virtual void writeOnStream(std::ostream& out);
    virtual void readFromStream(std::istream& in);

    //TilesVector generate(unsigned int inputSize, const Range& range);

protected:
    unsigned int computeComponentIndex(unsigned int i,double value);

private:
    void computeSize();

protected:
    std::vector<Range> ranges;
    std::vector<unsigned int> tilesN;
    unsigned int tilesSize;
};

class SelectiveTiles : public BasicTiles
{
public:
    SelectiveTiles(std::vector<unsigned int> stateComponents,
                   std::vector<Range>& ranges, std::vector<unsigned int>& tilesN);
    virtual unsigned int operator()(const arma::vec& input);
    virtual void writeOnStream(std::ostream& out);
    virtual void readFromStream(std::istream& in);

private:
    std::vector<unsigned int> stateComponents;
};



}


#endif /* INCLUDE_RELE_APPROXIMATORS_TILES_BASICTILES_H_ */
