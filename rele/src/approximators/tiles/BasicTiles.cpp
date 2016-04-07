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

#include "rele/approximators/tiles/BasicTiles.h"
#include "rele/utils/CSV.h"
#include <cassert>

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
    assert(ranges.size() == tilesN.size());
    computeSize();
}

unsigned int BasicTiles::operator()(const arma::vec& input)
{
    unsigned int multiplier = 1;
    unsigned int tileIndex = 0;

    for(unsigned int i = 0; i < ranges.size(); i++)
    {
        tileIndex += computeComponentIndex(i, input(i))*multiplier;
        multiplier *= tilesN[i];
    }

    return tileIndex;

}

unsigned int BasicTiles::computeComponentIndex(unsigned int i, double value)
{
    if(!ranges[i].contains(value) || value == ranges[i].hi())
        throw out_of_bounds();

    Range& range = ranges[i];
    unsigned int N = tilesN[i];
    return std::floor(N*(value - range.lo())/range.width());
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
    out << "Basic Tiles" << std::endl;
    out << "Ranges: ";
    CSVutils::vectorToCSV(ranges, out);
    out << "TilesN: ";
    CSVutils::vectorToCSV(tilesN, out);
}

void BasicTiles::readFromStream(std::istream& in)
{
    //TODO [SERIALIZATION] implement
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
    out << "Selective Tiles" << std::endl;
    out << "Ranges: ";
    CSVutils::vectorToCSV(ranges, out);
    out << "TilesN: ";
    CSVutils::vectorToCSV(tilesN, out);
    out << "State components: ";
    CSVutils::vectorToCSV(stateComponents, out);
}

void SelectiveTiles::readFromStream(std::istream& in)
{
    //TODO [SERIALIZATION] implement
}


}
