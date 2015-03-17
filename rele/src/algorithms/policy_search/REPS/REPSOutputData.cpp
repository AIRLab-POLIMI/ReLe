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

#include "policy_search/REPS/REPSOutputData.h"
#include "CSV.h"

using namespace std;
using namespace arma;

namespace ReLe
{

AbstractREPSOutputData::AbstractREPSOutputData(int N, double eps, const string& policyName, bool final) :
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
    os << "Using " << policyName << " policy"
       << endl << endl;
}

TabularREPSOutputData::TabularREPSOutputData(int N, double eps, const string& policyPrinted,
        bool final) : AbstractREPSOutputData(N, eps, "Tabular", final), policyPrinted(policyPrinted)
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


EpisodicREPSOutputData::EpisodicREPSOutputData(double eps,
        const string& policyName,
        const vec& policyParameters,
        const mat& policyVariance) :
    AgentOutputData(true), eps(eps), policyName(policyName),
    policyParameters(policyParameters), policyVariance(policyVariance)
{

}

void EpisodicREPSOutputData::writeData(ostream& os)
{

    os << "eps: " << eps << endl;
    os << policyName << endl;

    CSVutils::vectorToCSV(policyParameters, os);
    CSVutils::matrixToCSV(policyVariance, os);

}

void EpisodicREPSOutputData::writeDecoratedData(ostream& os)
{
    os << "- Parameters" << endl;
    os << "eps: " << eps << endl;
    os << "- Policy" << endl;
    os << "Using " << policyName << " policy"
       << endl << endl;
    os << "Policy parameters mean: " << policyParameters.t() << endl;
    os << "Policy parameters variance:" << endl << policyVariance << endl;
}

EpisodicREPSOutputData::~EpisodicREPSOutputData()
{

}

}
