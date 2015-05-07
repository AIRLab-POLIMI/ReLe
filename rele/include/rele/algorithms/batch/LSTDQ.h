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

#ifndef LSTDQ_H_
#define LSTDQ_H_

#include "Features.h"
#include "Transition.h"
#include "q_policy/e_Greedy.h"
#include "regressors/LinearApproximator.h"
#include <armadillo>
#include <cassert>

namespace ReLe
{

template<class ActionC>
class LSTDQ
{
public:
    LSTDQ(Dataset<ActionC, DenseState> data, e_GreedyApproximate& policy,
          Features_<arma::vec>& phi, double gamma) :
        data(data), Q(phi), policy(policy), gamma(gamma)
    {
        policy.setQ(&Q);
    }

    arma::vec run(bool firstTime)
    {
        /*** Initialize variables ***/
        int nbEpisodes = data.size();
        //compute the overall number of samples
        int nbSamples = 0;
        for (int k = 0; k < nbEpisodes; ++k)
            nbSamples += data[k].size();

        Features_<arma::vec>& basis = Q.getBasis();
        int df = basis.rows(); //number of features
        arma::mat PiPhihat(nbSamples, df, arma::fill::zeros);

        /*** Precompute Phihat and Rhat for all subsequent iterations ***/
        if (firstTime == true)
        {
            Phihat.zeros(nbSamples,df);
            Rhat.zeros(nbSamples);

            //scan the entire data set
            unsigned int idx = 0;
            for (auto episode : data)
            {
                for (auto tr : episode)
                {
                    //evaluate basis in input = [x; u]
                    arma::vec input = arma::join_vert(tr.x,tr.u);
                    arma::mat phi = basis(input);
                    Phihat.row(idx) = phi.t();
                    Rhat(idx) = tr.r[0];

                    // increment sample counter
                    ++idx;
                }
            }
        }

        /*** Loop through the samples ***/
        unsigned int idx = 0;
        for (auto episode : data)
        {
            for (auto tr : episode)
            {
                //Make sure the nextstate is not an absorbing state
                if (!tr.xn.isAbsorbing())
                {
                    /*** Compute the policy and the corresponding basis at the next state ***/
                    typename action_type<ActionC>::type nextAction = policy(tr.xn);
                    //evaluate basis in input = [x; u]
                    arma::vec input = arma::join_vert(tr.xn, nextAction);
                    arma::mat nextPhi = basis(input);
                    PiPhihat.row(idx) = nextPhi.t();

                    // increment sample counter
                    ++idx;
                }
            }
        }

        /*** Compute the matrices A and b ***/
        arma::mat A = Phihat.t() * (Phihat - gamma * PiPhihat);
        arma::vec b = Phihat.t() * Rhat;

        /*** Solve the system to find w ***/
        arma::vec w;
        int rank = arma::rank(A);
        if (rank == df)
        {
            w = arma::solve(A,b);
        }
        else
        {
            w = arma::pinv(A)*b;
        }

        return w;
    }

    LinearApproximator& getQ()
    {
        return Q;
    }

protected:
    Dataset<ActionC, DenseState>& data;
    LinearApproximator Q;
    e_GreedyApproximate& policy;
    double gamma;
    arma::mat Phihat;
    arma::vec Rhat;

};

}//end namespace

#endif //LSTDQ_H_
