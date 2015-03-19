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

#include "ConsoleManager.h"

#include <iostream>
#include <sstream>

using namespace std;

namespace ReLe
{

ConsoleManager::ConsoleManager(unsigned int max, unsigned int step, bool percentage) :
    max(max), step(step), percentage(percentage), lastPrintSize(0)
{

}

void ConsoleManager::printProgress(unsigned int progress)
{

    if(progress % step == 0)
    {
        stringstream ss;

        if(percentage)
        {
            double pValue = static_cast<double>(progress)/static_cast<double>(max);

            ss << "# " << pValue << "% ";
        }
        else
        {
            ss << "# " << progress << "/" << max << " ";
        }

        if(lastPrintSize != 0)
            cout << string('\b', lastPrintSize) << string(' ', lastPrintSize) << '\r';
        lastPrintSize = ss.str().size();
        cout << ss.str();
        cout.flush();
    }
}




}
