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

#include "policy_search/REPS/REPSOutputData.h"
#include "CSV.h"

using namespace std;
using namespace arma;

namespace ReLe
{

AbstractREPSOutputData::AbstractREPSOutputData(int N, double eps,
        const string& policyName, bool final) :
    AgentOutputData(final), N(N), eps(eps), policyName(policyName)
{

}

AbstractREPSOutputData::~AbstractREPSOutputData()
{

}

void AbstractREPSOutputData::writeInfo(ostream& os)
{
    os << "eps: " << eps << endl;
    os << "N: " << N << endl;
    os << policyName << endl;
}

void AbstractREPSOutputData::writeDecoratedInfo(ostream& os)
{
    os << "- Parameters" << endl;
    os << "eps: " << eps << endl;
    os << "N: " << N << endl;
    os << "- Policy" << endl;
    os << "Using " << policyName << " policy" << endl << endl;
}

TabularREPSOutputData::TabularREPSOutputData(int N, double eps,
        const string& policyPrinted, bool final) :
    AbstractREPSOutputData(N, eps, "Tabular", final),
    policyPrinted(policyPrinted)
{

}

void TabularREPSOutputData::writeData(ostream& os)
{
    writeInfo(os);
    os << policyPrinted << endl; //TODO change this
}

void TabularREPSOutputData::writeDecoratedData(ostream& os)
{
    writeDecoratedInfo(os);
    os << policyPrinted << endl;
}

TabularREPSOutputData::~TabularREPSOutputData()
{

}

REPSOutputData::REPSOutputData(unsigned int nbIndividual,
                               unsigned int nbParams, unsigned int nbEvals) :
    BlackBoxOutputData(nbIndividual, nbParams, nbEvals)
{
    eta = 0;
}

void REPSOutputData::writeData(ostream& os)
{
    os << metaParams.n_elem << endl;
    CSVutils::vectorToCSV(metaParams, os);

    os << individuals[0].Pparams.n_elem << endl;
    os << individuals[0].Jvalues.n_elem << endl;
    os << individuals.size() << endl;

    for (auto& individual : individuals)
    {
        CSVutils::vectorToCSV(individual.Pparams, os);
        CSVutils::vectorToCSV(individual.Jvalues, os);
    }

    os << eta << std::endl;
    CSVutils::matrixToCSV(covariance, os);
}

void REPSOutputData::writeDecoratedData(ostream& os)
{
    //TODO implement
    writeData(os);
}

}
