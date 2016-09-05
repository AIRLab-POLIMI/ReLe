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

#include "rele/utils/ModularRange.h"
#include <cmath>
#include <stdexcept>

using namespace std;
using namespace ReLe;

int main()
{
    cout << "Modular Range Test" << endl;

    RangePi range1;
    Range2Pi range2;

    for(int k = -2; k < 3; k++)
        for(int i = 0; i < 8; i++)
        {
            double angle = i*M_PI/4.0;
            double anglePi = range1.wrap(angle + 2 * k * M_PI);
            double angle2Pi = range2.wrap(angle  + 2 * k * M_PI);
            if(-M_PI > anglePi && anglePi > M_PI)
                throw std::logic_error("Bug in RangePi::wrap.");
            if(0 > angle2Pi && angle2Pi > 2 * M_PI)
                throw std::logic_error("Bug in Range2Pi::wrap.");
            cout << "angle 2PI: " << angle << " = " << angle2Pi << endl;
            cout << "angle PI:  " << (angle > M_PI ? angle - 2*M_PI : angle)  << " = "  << anglePi << endl;
        }
}
