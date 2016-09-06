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

#include "rele/core/callbacks/CoreCallback.h"

#include <iostream>

using namespace std;



namespace ReLe
{


CoreCallback::~CoreCallback()
{

}

void CoreProgressBar::run(unsigned int step)
{
    double ratio = static_cast<double>(step)/static_cast<double>(stepNumbers);
    const unsigned int size = 100;
    unsigned int done = size*ratio;
    unsigned int todo = size - done;

    cout << "\e[?25l\r[" << string(done, '#') << string(todo, ' ') << "] " << done << "%";
}

void CoreProgressBar::runEnd()
{
    cout << "\r[" << string(100, '#') << "] " << 100 << "%\e[?25h";
    cout << endl;
}


}
