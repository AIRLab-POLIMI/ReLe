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

#include "rele/algorithms/td/R-Learning.h"
#include "rele/utils/RandomGenerator.h"

#include "rele/utils/CSV.h"

using namespace std;
using namespace arma;

namespace ReLe
{

R_Learning::R_Learning(ActionValuePolicy<FiniteState>& policy, LearningRate& alpha, LearningRate& beta) :
    FiniteTD(policy, alpha), beta(beta)
{
    ro = 0;
}

void R_Learning::initEpisode(const FiniteState& state, FiniteAction& action)
{
    sampleAction(state, action);
}

void R_Learning::sampleAction(const FiniteState& state, FiniteAction& action)
{
    x = state.getStateN();
    u = policy(x);

    action.setActionN(u);
}

void R_Learning::step(const Reward& reward, const FiniteState& nextState,
                      FiniteAction& action)
{
    size_t xn = nextState.getStateN();
    double r = reward[0];
    double maxQxn, maxQx;

    const rowvec& Qxn = Q.row(xn);
    maxQxn = Qxn.max();

    //Update value function
    double delta = r - ro  + maxQxn - Q(x, u);
    Q(x, u) = Q(x, u) + alpha(x, u) * delta;

    const rowvec& Qx = Q.row(x);
    maxQx = Qx.max();

    //update mean reward
    if(Q(x, u) == maxQx)
    {
        double deltaRo = r - ro + maxQxn - maxQx;
        ro = ro + beta(x, u) * deltaRo;
    }

    //update action and state
    x = xn;
    u = policy(xn);

    //set next action
    action.setActionN(u);
}

void R_Learning::endEpisode(const Reward& reward)
{
    //Last update
    double r = reward[0];
    double delta = r - Q(x, u);
    Q(x, u) = Q(x, u) + alpha(x, u) * delta;


    double maxQx;
    const rowvec& Qx = Q.row(x);
    maxQx = Qx.max();

    if(Q(x, u) == maxQx)
    {
        double deltaRo = r - ro - maxQx;
        ro = ro + beta(x, u) * deltaRo;
    }
}

R_Learning::~R_Learning()
{

}


R_LearningOutput::R_LearningOutput(const std::string& alpha,
                                   const std::string& beta,
                                   const std::string& policyName,
                                   const hyperparameters_map& policyHPar,
                                   const arma::mat& Q,
                                   double ro)
    : FiniteTDOutput(1.0, alpha, policyName, policyHPar, Q) , beta(beta), ro(ro)
{

}

void R_LearningOutput::writeData(std::ostream& os)
{
    os << policyName << endl;
    os << "gamma: " << gamma << endl;
    os << "alpha: " << alpha << endl;
    os << "beta: " << beta << endl;
    os << policyHPar << endl;

    CSVutils::matrixToCSV(Q, os);
    os << ro << endl;
}

void R_LearningOutput::writeDecoratedData(std::ostream& os)
{
    os << "Using " << policyName << " policy"
       << endl << endl;

    os << "- Parameters" << endl;
    os << "gamma: " << gamma << endl;
    os << "alpha: " << alpha << endl;
    os << "beta: " << beta << endl;
    os << policyHPar << endl;

    os << "- Action-value function" << endl;
    for (unsigned int i = 0; i < Q.n_rows; i++)
        for (unsigned int j = 0; j < Q.n_cols; j++)
        {
            os << "Q(" << i << ", " << j << ") = " << Q(i, j) << endl;
        }

    os << "Mean Reward: " << ro << endl;

    os << "- Policy" << endl;
    for (unsigned int i = 0; i < Q.n_rows; i++)
    {
        arma::uword policy;
        Q.row(i).max(policy);
        os << "policy(" << i << ") = " << policy << endl;
    }
}


}
