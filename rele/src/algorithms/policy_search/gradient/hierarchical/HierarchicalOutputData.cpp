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

#include "HierarchicalOutputData.h"
#include "CSV.h"

#include <cassert>

using namespace std;

namespace ReLe
{

HierarchicalOutputData::HierarchicalOutputData()
{
    bottomReached = true;
}


void HierarchicalOutputData::writeData(std::ostream& os)
{
	os << traces.size() << endl;

	for(auto& trace : traces)
	{
		os << trace.size() << ",";
		CSVutils::vectorToCSV(trace, os);
	}

}

void HierarchicalOutputData::writeDecoratedData(std::ostream& os)
{
	//TODO decorated
	writeData(os);
}

void HierarchicalOutputData::addOptionCall(unsigned int option)
{
	if(bottomReached)
	{
		assert(traces.size() == 0 || traces.back().size() > 0);
		traces.resize(traces.size() + 1);
		traceCount.resize(traces.size() + 1, 0);
		bottomReached = false;
	}

	traces.back().push_back(option);
}

void HierarchicalOutputData::addLowLevelCommand()
{
	bottomReached = true;
	traceCount.back()++;
}





}
