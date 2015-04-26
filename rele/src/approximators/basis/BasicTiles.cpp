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

#include "tiles/BasicTiles.h"

namespace ReLe
{

//////////////////////////////
// Basic Tiles
//////////////////////////////

BasicTiles::BasicTiles(Range& range, unsigned int tilesN)
{
    this->tilesSize = tilesN;
    this->ranges.push_back(range);
    this->tilesN.push_back(tilesN);
}

BasicTiles::BasicTiles(std::vector<Range>& ranges, std::vector<unsigned int>& tilesN)
    : ranges(ranges), tilesN(tilesN)
{
    computeSize();
}

unsigned int BasicTiles::operator()(const arma::vec& input)
{
    unsigned int multiplier = 1;
    unsigned int tileIndex = 0;

    for(unsigned int i = 0; i < input.n_elem; i++)
    {
        tileIndex += computeComponentIndex(i, input(i))*multiplier;
        multiplier *= tilesN[i];
    }

    return tileIndex;

}

unsigned int BasicTiles::computeComponentIndex(unsigned int i, double value)
{
    //TODO add offset to tiles
    Range& range = ranges[i];
    unsigned int N = tilesN[i];
    return std::floor(N*(value - range.Lo())/range.Width());
}

void BasicTiles::computeSize()
{
    tilesSize = 1;

    for (auto c : tilesN)
    {
        tilesSize *= c;
    }
}

void BasicTiles::writeOnStream(std::ostream& out)
{
    //TODO writeonstream
}

void BasicTiles::readFromStream(std::istream& in)
{
    //TODO readfromstream
}

//////////////////////////////
// Selective Tiles
//////////////////////////////

SelectiveTiles::SelectiveTiles(std::vector<unsigned int> stateComponents,
                               std::vector<Range>& ranges, std::vector<unsigned int>& tilesN)
    : BasicTiles(ranges, tilesN), stateComponents(stateComponents)
{

}

unsigned int SelectiveTiles::operator()(const arma::vec& input)
{
    unsigned int multiplier = 1;
    unsigned int tileIndex = 0;

    for(unsigned int i = 0; i < stateComponents.size(); i++)
    {
        unsigned int index = stateComponents[i];
        tileIndex += computeComponentIndex(i, input(index))*multiplier;
        multiplier *= tilesN[i];
    }

    return tileIndex;
}

void SelectiveTiles::writeOnStream(std::ostream& out)
{
    //TODO writeonstream
}

void SelectiveTiles::readFromStream(std::istream& in)
{
    //TODO readfromstream
}


}
