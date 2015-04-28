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

#include "Transition.h"
#include "RandomGenerator.h"

#include <sstream>

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{
    unsigned int episodes = 5;
    unsigned int transitions = 6;

    Dataset<FiniteAction, FiniteState> dataset;

    for(unsigned int i = 0; i < episodes; i++)
    {
        Episode<FiniteAction, FiniteState> episode;
        bool first = true;
        unsigned int state = 0;
        for(unsigned int j = 0; j < transitions; j++)
        {
            Transition<FiniteAction, FiniteState> transition;

            FiniteState x(state);
            FiniteAction u(RandomGenerator::randu32());
            state = RandomGenerator::randu32();
            FiniteState xn(state);
            Reward r(1);
            r[0] = RandomGenerator::randu32();

            transition.x = x;
            transition.u = u;
            transition.r = r;
            transition.xn = xn;

            episode.push_back(transition);
        }

        dataset.push_back(episode);
    }

    stringstream ss;
    string before;
    string after;

    dataset.writeToStream(ss);
    before = ss.str();

    Dataset<FiniteAction, FiniteState> datasetReloaded;
    datasetReloaded.readFromStream(ss);

    //Clear stream
    ss.str("");
    ss.clear();

    datasetReloaded.writeToStream(ss);
    after = ss.str();

    cout << "---------------------" << endl;
    cout << before;
    cout << "---------------------" << endl;
    cout << after;
    cout << "---------------------" << endl;

    if(before == after)
    {
        cout << "Test passed!" << endl;
        cout << "example: " << endl;
        Transition<FiniteAction, FiniteState>& t = dataset[0][0];
        Transition<FiniteAction, FiniteState>& tr = datasetReloaded[0][0];
        cout << "Dataset" << endl;
        cout << "x = " << t.x << endl;
        cout << "u = " << t.u << endl;
        cout << "r = " << t.r << endl;
        cout << "xn = " << t.xn << endl;

        cout << "Dataset reloaded" << endl;
        cout << "x = " << tr.x << endl;
        cout << "u = " << tr.u << endl;
        cout << "r = " << tr.r << endl;
        cout << "xn = " << tr.xn << endl;
    }
    else
        cout << "Test failed!" << endl;
}
