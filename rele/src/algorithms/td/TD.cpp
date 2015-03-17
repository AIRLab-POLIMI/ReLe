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

#include "td/TD.h"
#include "CSV.h"

using namespace std;
using namespace arma;

namespace ReLe
{

FiniteTD::FiniteTD(ActionValuePolicy<FiniteState>& policy) :
    policy(policy)
{
    x = 0;
    u = 0;

    //Default algorithm parameters
    alpha = 0.2;
}

void FiniteTD::endEpisode()
{

}

void FiniteTD::init()
{
    Q.zeros(task.finiteStateDim, task.finiteActionDim);
    policy.setQ(&Q);
    policy.setNactions(task.finiteActionDim);
}


FiniteTDOutput::FiniteTDOutput(double gamma,
                               double alpha,
                               string policyName,
                               string policyHPar,
                               mat Q) :
    AgentOutputData(true),gamma(gamma), alpha(alpha),
    policyName(policyName),policyHPar(policyHPar), Q(Q)
{

}

void FiniteTDOutput::writeData(ostream& os)
{
    os << policyName << endl;
    os << "gamma: " << gamma << endl;
    os << "alpha: " << alpha << endl;
    os << policyHPar; //TODO change this

    CSVutils::matrixToCSV(Q, os);

}

void FiniteTDOutput::writeDecoratedData(ostream& os)
{
    os << "Using " << policyName << " policy"
       << endl << endl;

    os << "- Parameters" << endl;
    os << "gamma: " << gamma << endl;
    os << "alpha: " << alpha << endl;
    os << policyHPar;

    os << "- Action-value function" << endl;
    for (unsigned int i = 0; i < Q.n_rows; i++)
        for (unsigned int j = 0; j < Q.n_cols; j++)
        {
            os << "Q(" << i << ", " << j << ") = " << Q(i, j) << endl;
        }

    os << "- Policy" << endl;
    for (unsigned int i = 0; i < Q.n_rows; i++)
    {
        unsigned int policy;
        Q.row(i).max(policy);
        os << "policy(" << i << ") = " << policy << endl;
    }
}

LinearTD::LinearTD(ActionValuePolicy<DenseState>& policy,
                   LinearApproximator& la) :
    Q(la), policy(policy)
{
    u = 0;

    //Default parameters
    alpha = 0.2;
}

void LinearTD::endEpisode()
{

}

void LinearTD::init()
{
    x.zeros(task.continuosStateDim);
    policy.setQ(&Q);
    policy.setNactions(task.finiteActionDim);
}

LinearTDOutput::LinearTDOutput(double gamma,
                               double alpha,
                               string policyName,
                               string policyHPar,
                               vec Qw) :
    AgentOutputData(true),gamma(gamma), alpha(alpha),
    policyName(policyName),policyHPar(policyHPar), Qw(Qw)
{

}

void LinearTDOutput::writeData(ostream& os)
{
    os << policyName << endl;
    os << "gamma: " << gamma << endl;
    os << "alpha: " << alpha << endl;
    os << policyHPar; //TODO change this

    CSVutils::vectorToCSV(Qw, os);

}

void LinearTDOutput::writeDecoratedData(ostream& os)
{
    os << "Using " << policyName << " policy"
       << endl << endl;

    os << "- Parameters" << endl ;
    os << "gamma: " << gamma << endl;
    os << "alpha: " << alpha << endl;
    os << policyHPar;

    os << "- Action-value function" << endl;
    os << Qw.t() << endl;

}

}
