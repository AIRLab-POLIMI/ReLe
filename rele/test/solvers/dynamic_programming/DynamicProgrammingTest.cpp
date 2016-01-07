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


#include "rele/solvers/dynamic_programming/BasicDynamicProgramming.h"
#include "rele/core/FiniteMDP.h"
#include "rele/generators/SimpleChainGenerator.h"



using namespace ReLe;
using namespace std;

int main(int argc, char *argv[])
{
    /* Create simple chain and optimal policy */
    SimpleChainGenerator generator;
    generator.generate(5, 2);

    FiniteMDP mdp = generator.getMDP(0.9);

    ValueIteration solver1(mdp, 0.01);
    PolicyIteration solver2(mdp);

    solver1.solve();
    solver2.solve();


    cout << "Value iteration results:" << endl;
    cout << solver1.getPolicy().printPolicy();

    cout << "Policy iteration results:" << endl;
    cout << solver2.getPolicy().printPolicy();

    return 0;
}
