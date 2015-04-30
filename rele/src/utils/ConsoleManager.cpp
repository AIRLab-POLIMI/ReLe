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

#include "ConsoleManager.h"

#include <iostream>
#include <sstream>

using namespace std;

namespace ReLe
{

ConsoleManager::ConsoleManager(unsigned int max, unsigned int step, bool percentage) :
    max(max), step(step), percentage(percentage)
{

}

void ConsoleManager::printInfo(const string& info)
{
    cout << "# " << info << endl;
}

void ConsoleManager::printProgress(unsigned int progress)
{

    if(progress % step == 0)
    {
        stringstream ss;

        if(percentage)
        {
            double pValue = static_cast<double>(progress + 1)/static_cast<double>(max)*100.0;

            ss << "# " << pValue << "% ";
        }
        else
        {
            ss << "# " << progress + 1 << "/" << max << " ";
        }

        cout << '\r';
        cout << ss.str();
        cout.flush();

        if(progress + 1 == max)
            cout << endl;
    }
}




}
