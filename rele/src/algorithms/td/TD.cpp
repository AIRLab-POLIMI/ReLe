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

#include "rele/algorithms/td/TD.h"
#include "rele/utils/CSV.h"

using namespace std;
using namespace arma;

namespace ReLe
{

FiniteTD::FiniteTD(ActionValuePolicy<FiniteState>& policy, LearningRate& alpha) :
    policy(policy), alpha(alpha)
{
    x = 0;
    u = 0;
}

void FiniteTD::endEpisode()
{

}

void FiniteTD::init()
{
    Q.zeros(task.statesNumber, task.actionsNumber);
    policy.setQ(&Q);
    policy.setNactions(task.actionsNumber);
}


FiniteTDOutput::FiniteTDOutput(double gamma,
                               const std::string& alpha,
                               const std::string& policyName,
                               const hyperparameters_map& policyHPar,
                               const mat& Q) :
    AgentOutputData(true),gamma(gamma), alpha(alpha),
    policyName(policyName),policyHPar(policyHPar), Q(Q)
{

}

void FiniteTDOutput::writeData(ostream& os)
{
    os << policyName << endl;
    os << "gamma: " << gamma << endl;
    os << "alpha: " << alpha << endl;
    os << policyHPar << endl;

    CSVutils::matrixToCSV(Q, os);

}

void FiniteTDOutput::writeDecoratedData(ostream& os)
{
    os << "Using " << policyName << " policy"
       << endl << endl;

    os << "- Parameters" << endl;
    os << "gamma: " << gamma << endl;
    os << "alpha: " << alpha << endl;
    os << policyHPar << endl;

    os << "- Action-value function" << endl;
    for (unsigned int i = 0; i < Q.n_rows; i++)
        for (unsigned int j = 0; j < Q.n_cols; j++)
        {
            os << "Q(" << i << ", " << j << ") = " << Q(i, j) << endl;
        }

    os << "- Policy" << endl;
    for (unsigned int i = 0; i < Q.n_rows; i++)
    {
        arma::uword policy;
        Q.row(i).max(policy);
        os << "policy(" << i << ") = " << policy << endl;
    }
}

LinearTD::LinearTD(Features& phi,
                   ActionValuePolicy<DenseState>& policy,
                   LearningRateDense& alpha) :
    Q(phi), policy(policy), alpha(alpha)
{
    u = 0;
}

void LinearTD::endEpisode()
{

}

void LinearTD::init()
{
    x.zeros(task.stateDimensionality);
    policy.setQ(&Q);
    policy.setNactions(task.actionsNumber);
}

LinearTDOutput::LinearTDOutput(double gamma,
                               const std::string& alpha,
                               const std::string& policyName,
                               const hyperparameters_map& policyHPar,
                               const arma::vec Qw) :
    AgentOutputData(true),gamma(gamma), alpha(alpha),
    policyName(policyName),policyHPar(policyHPar), Qw(Qw)
{

}

void LinearTDOutput::writeData(ostream& os)
{
    os << policyName << endl;
    os << "gamma: " << gamma << endl;
    os << "alpha: " << alpha << endl;
    os << policyHPar << endl;

    CSVutils::vectorToCSV(Qw, os);

}

void LinearTDOutput::writeDecoratedData(ostream& os)
{
    os << "Using " << policyName << " policy"
       << endl << endl;

    os << "- Parameters" << endl ;
    os << "gamma: " << gamma << endl;
    os << "alpha: " << alpha << endl;
    os << policyHPar  << endl;

    os << "- Action-value function" << endl;
    os << Qw.t() << endl;

}

}
