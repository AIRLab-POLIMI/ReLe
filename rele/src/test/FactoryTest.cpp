/*
 * rele,
 *
 *
 * Copyright (C) 2015  Davide Tateo & Matteo Pirotta
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

#include "policy_search/NES/NES.h"
#include "policy_search/REPS/REPS.h"
#include "DifferentiableNormals.h"
#include "Core.h"
#include "parametric/differentiable/GibbsPolicy.h"
#include "BasisFunctions.h"
#include "basis/PolynomialFunction.h"
#include "basis/ConditionBasedFunction.h"
#include "RandomGenerator.h"
#include "FileManager.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>
#include "Environment.h"
#include "DeepSeaTreasure.h"
#include "NLS.h"

using namespace std;
using namespace ReLe;
using namespace arma;


#define CREATE_MDP(ActionC, StateC, environment) \
Environment<ActionC, StateC>* mdp = \
static_cast<Environment<ActionC, StateC>*>(MDPFactory::init(environment));

class MDPFactory
{
public:

    static void* init(const std::string prova)
    {
        if(prova == "deep")
        {
            return new DeepSeaTreasure();
        }
        else
        {
            return nullptr;
        }

    }

};

int main(int argc, char *argv[])
{
    std::string environment = "deep";
    //Environment<FiniteAction, DenseState>* mdp = static_cast<Environment<FiniteAction, DenseState>*>(MDPFactory::init(environment));
    CREATE_MDP(FiniteAction, DenseState, environment)

    FiniteAction a;
    DenseState s;
    Reward r(1);

    a.setActionN(0);

    mdp->getInitialState(s);
    cout << "s: " << s << endl;

    mdp->step(a, s, r);
    cout << "s: " << s << " r: " << r << endl;

    return 0;
}
