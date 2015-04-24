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

BasicTiles::BasicTiles(std::vector<Range>& ranges, std::vector<unsigned int>& tilesN)
	: ranges(ranges), tilesN(tilesN)
{

}

unsigned int BasicTiles::operator()(const arma::vec& input)
{

}

void BasicTiles::writeOnStream(std::ostream& out)
{

}

void BasicTiles::readFromStream(std::istream& in)
{

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

}

void SelectiveTiles::writeOnStream(std::ostream& out)
{

}

void SelectiveTiles::readFromStream(std::istream& in)
{

}


}
