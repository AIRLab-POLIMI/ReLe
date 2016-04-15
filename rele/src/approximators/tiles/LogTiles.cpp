/*
 * rele,
 *
 *
 * Copyright (C) 2016 Davide Tateo
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

#include "rele/approximators/tiles/LogTiles.h"

namespace ReLe
{

LogTiles::LogTiles(const Range& range, unsigned int tilesN)
    : BasicTiles(computeLogRange(range), tilesN), minComponents(1)
{
	 minComponents(0) = range.lo();
}

LogTiles::LogTiles(const std::vector<Range>& ranges,
			const std::vector<unsigned int>& tilesN)
    : BasicTiles(computeLogRange(ranges), tilesN)
{

}

unsigned int LogTiles::operator()(const arma::vec& input)
{
	arma::vec inputHat = arma::log(input - minComponents + 1);
	return BasicTiles::operator()(inputHat);
}

void LogTiles::writeOnStream(std::ostream& out)
{
    //TODO [SERIALIZATION] implement
}

void LogTiles::readFromStream(std::istream& in)
{
    //TODO [SERIALIZATION] implement
}

Range LogTiles::computeLogRange(const Range& range)
{
	return Range(0, std::log(range.width() + 1));
}

std::vector<Range> LogTiles::computeLogRange(const std::vector<Range>& range)
{
	minComponents.set_size(range.size());
    std::vector<Range> logRanges;
    for(unsigned int i = 0; i < range.size(); i++)
    {
        logRanges.push_back(computeLogRange(range[i]));
        minComponents(i) = range[i].lo();
    }

    return logRanges;
}


CenteredLogTiles::CenteredLogTiles(const Range& range, unsigned int tilesN)
	: BasicTiles(computeLogRange(range), tilesN), centers(1)
{
	centers(0) = range.mid();
}

CenteredLogTiles::CenteredLogTiles(const std::vector<Range>& ranges,
			const std::vector<unsigned int>& tilesN)
	: BasicTiles(computeLogRange(ranges), tilesN)
{

}

unsigned int CenteredLogTiles::operator()(const arma::vec& input)
{
	arma::vec delta = input - centers;
	arma::vec inputHat = arma::sign(delta) % arma::log(arma::abs(delta) + 1.0);
	return BasicTiles::operator()(inputHat);
}

void CenteredLogTiles::writeOnStream(std::ostream& out)
{
	//TODO [SERIALIZATION] implement
}

void CenteredLogTiles::readFromStream(std::istream& in)
{
	//TODO [SERIALIZATION] implement
}

Range CenteredLogTiles::computeLogRange(const Range& range)
{
	double delta = std::log(range.width()/2.0 + 1.0);
	return Range(-delta, delta);
}

std::vector<Range> CenteredLogTiles::computeLogRange(const std::vector<Range>& range)
{
	centers.set_size(range.size());
    std::vector<Range> logRanges;
    for(unsigned int i = 0; i < range.size(); i++)
    {
        logRanges.push_back(computeLogRange(range[i]));
        centers(i) = range[i].mid();
    }

    return logRanges;
}




}
