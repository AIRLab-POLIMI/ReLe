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
class LSTDQ_
{
public:

    LSTDQ_(Dataset<ActionC, DenseState>& data, e_GreedyApproximate& policy,
           Features_<arma::vec>& phi, double gamma) :
        data(data), Q(phi), policy(policy), gamma(gamma)
    {
        policy.setQ(&Q);
    }

    LinearApproximator& getQ()
    {
        return Q;
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
        arma::mat A(df, df, arma::fill::zeros);
        arma::vec b(df, arma::fill::zeros);

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
                    arma::vec s = tr.x;
                    arma::vec a(1);
                    a(0) = tr.u;
                    arma::vec input = arma::join_vert(s, a);
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
                    arma::vec s = tr.x;
                    arma::vec a(1);
                    a(0) = nextAction;
                    arma::vec input = arma::join_vert(s, a);
                    arma::mat nextPhi = basis(input);
                    PiPhihat.row(idx) = nextPhi.t();

                    // increment sample counter
                    ++idx;
                }
            }
        }

        /*** Compute the matrices A and b ***/
        //        arma::mat A = Phihat.t() * (Phihat - gamma * PiPhihat);
        //        arma::vec b = Phihat.t() * Rhat;
        computeAandB(PiPhihat, A, b);

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

    virtual void computeAandB(const arma::mat& PiPhihat, arma::mat& A, arma::vec& b) = 0;

    virtual ~LSTDQ_()
    {

    }

protected:
    Dataset<ActionC, DenseState>& data;
    LinearApproximator Q;
    e_GreedyApproximate& policy;
    double gamma;
    arma::mat Phihat;
    arma::vec Rhat;
};

/**
 * Least-Squares Policy Iteration
 * Michail G. Lagoudakis and Ronald Parr
 * Journal of Machine Learning Research, 4, 2003, pp. 1107-1149.
 * Source code: https://www.cs.duke.edu/research/AI/LSPI/lspi.tar.gz
 */
template<class ActionC>
class LSTDQ : public LSTDQ_<ActionC>
{
    typedef LSTDQ_<ActionC> Base;
//    using Base::data;
//    using Base::Q;
//    using Base::policy;
    using Base::gamma;
    using Base::Phihat;
    using Base::Rhat;

public:
    LSTDQ(Dataset<ActionC, DenseState>& data, e_GreedyApproximate& policy,
          Features_<arma::vec>& phi, double gamma)
        : LSTDQ_<ActionC>(data, policy, phi, gamma)
    {
    }

    virtual void computeAandB(const arma::mat& PiPhihat, arma::mat& A, arma::vec& b)
    {
        A = Phihat.t() * (Phihat - gamma * PiPhihat);
        b = Phihat.t() * Rhat;
    }

    virtual ~LSTDQ()
    {

    }

    //    arma::vec run(bool firstTime)
    //    {
    //        /*** Initialize variables ***/
    //        int nbEpisodes = data.size();
    //        //compute the overall number of samples
    //        int nbSamples = 0;
    //        for (int k = 0; k < nbEpisodes; ++k)
    //            nbSamples += data[k].size();

    //        Features_<arma::vec>& basis = Q.getBasis();
    //        int df = basis.rows(); //number of features
    //        arma::mat PiPhihat(nbSamples, df, arma::fill::zeros);

    //        /*** Precompute Phihat and Rhat for all subsequent iterations ***/
    //        if (firstTime == true)
    //        {
    //            Phihat.zeros(nbSamples,df);
    //            Rhat.zeros(nbSamples);

    //            //scan the entire data set
    //            unsigned int idx = 0;
    //            for (auto episode : data)
    //            {
    //                for (auto tr : episode)
    //                {
    //                    //evaluate basis in input = [x; u]
    //                    arma::vec input = arma::join_vert(tr.x,tr.u);
    //                    arma::mat phi = basis(input);
    //                    Phihat.row(idx) = phi.t();
    //                    Rhat(idx) = tr.r[0];

    //                    // increment sample counter
    //                    ++idx;
    //                }
    //            }
    //        }

    //        /*** Loop through the samples ***/
    //        unsigned int idx = 0;
    //        for (auto episode : data)
    //        {
    //            for (auto tr : episode)
    //            {
    //                //Make sure the nextstate is not an absorbing state
    //                if (!tr.xn.isAbsorbing())
    //                {
    //                    /*** Compute the policy and the corresponding basis at the next state ***/
    //                    typename action_type<ActionC>::type nextAction = policy(tr.xn);
    //                    //evaluate basis in input = [x; u]
    //                    arma::vec input = arma::join_vert(tr.xn, nextAction);
    //                    arma::mat nextPhi = basis(input);
    //                    PiPhihat.row(idx) = nextPhi.t();

    //                    // increment sample counter
    //                    ++idx;
    //                }
    //            }
    //        }

    //        /*** Compute the matrices A and b ***/
    //        arma::mat A = Phihat.t() * (Phihat - gamma * PiPhihat);
    //        arma::vec b = Phihat.t() * Rhat;

    //        /*** Solve the system to find w ***/
    //        arma::vec w;
    //        int rank = arma::rank(A);
    //        if (rank == df)
    //        {
    //            w = arma::solve(A,b);
    //        }
    //        else
    //        {
    //            w = arma::pinv(A)*b;
    //        }

    //        return w;
    //    }
};

/**
 * Least-Squares Policy Iteration
 * Michail G. Lagoudakis and Ronald Parr
 * Journal of Machine Learning Research, 4, 2003, pp. 1107-1149.
 * Source code: https://www.cs.duke.edu/research/AI/LSPI/lspi.tar.gz
 */
template<class ActionC>
class LSTDQBe : public LSTDQ_<ActionC>
{
    typedef LSTDQ_<ActionC> Base;
//    using Base::data;
//    using Base::Q;
//    using Base::policy;
    using Base::gamma;
    using Base::Phihat;
    using Base::Rhat;

public:
    LSTDQBe(Dataset<ActionC, DenseState>& data, e_GreedyApproximate& policy,
            Features_<arma::vec>& phi, double gamma)
        : LSTDQ_<ActionC>(data, policy, phi, gamma)
    {
    }

    // LSTDQ_ interface
    void computeAandB(const arma::mat& PiPhihat, arma::mat &A, arma::vec &b)
    {
        /*** Compute the matrices A and b ***/
        arma::mat tmp = Phihat - gamma * PiPhihat;
        A = tmp.t() * tmp;
        b = tmp.t() * Rhat;
    }

};

}//end namespace

#endif //LSTDQ_H_
