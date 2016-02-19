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

#include "rele/approximators/Features.h"
#include "rele/core/Transition.h"
#include "rele/policy/q_policy/e_Greedy.h"
#include "rele/approximators/regressors/others/LinearApproximator.h"
#include <armadillo>
#include <cassert>
#include "rele/core/BatchAgent.h"

namespace ReLe
{

//#define REMOVE_LAST

template<class ActionC>
class LSTDQ_
{
public:

    LSTDQ_(Dataset<ActionC, DenseState>& data, e_GreedyApproximate& policy,
           Features_<arma::vec>& phi, double gamma) :
        data(data), Q(phi), policy(policy), gamma(gamma)
    {
        policy.setQ(&Q);
        //        arma::vec xi = {0,0.2,0};
        //        Features_<arma::vec>& rr = Q.getBasis();
        //        std::cout << rr(xi) << std::endl;
        //        std::cout << std::endl;
    }

    LinearApproximator& getQ()
    {
        return Q;
    }

    arma::vec run(bool firstTime)
    {
        /*** Initialize variables ***/
        int nbEpisodes = this->data.size();
        //compute the overall number of samples
        int nbSamples = 0;
        for (int k = 0; k < nbEpisodes; ++k)
            nbSamples += this->data[k].size();

        Features_<arma::vec>& basis = Q.getFeatures();
        int df = basis.rows(); //number of features
        arma::mat PiPhihat(nbSamples, df, arma::fill::zeros);
        arma::mat A(df, df, arma::fill::zeros);
        arma::vec b(df, arma::fill::zeros);

        /*** Precompute Phihat and Rhat for all subsequent iterations ***/
        if (firstTime == true)
        {
            Phihat.set_size(nbSamples,df);
            Rhat.set_size(nbSamples);

            //scan the entire data set
            arma::vec a(1);
            arma::mat phi;
            unsigned int idx = 0;
            for (auto episode : this->data)
            {
                for (auto tr : episode)
                {
                    //compute basis in current state-action pair
                    a(0) = tr.u;
                    arma::vec& tmp = tr.x;
                    arma::vec input = arma::join_vert(tmp, a);
                    phi = basis(input);

                    //update matricies
                    Phihat.row(idx) = phi.t();
                    Rhat(idx) = tr.r[0];

                    // increment sample counter
                    ++idx;
                }
            }
        }

        /*** Loop through the samples ***/
        unsigned int idx = 0;
        arma::vec a(1), input;
        arma::mat nextPhi;
        for (auto episode : this->data)
        {
            for (auto tr : episode)
            {
                //Make sure the nextstate is not an absorbing state
#ifdef REMOVE_LAST
                if (!tr.xn.isAbsorbing())
                {
#endif
                    /*** Compute the policy and the corresponding basis at the next state ***/
                    typename action_type<ActionC>::type nextAction = policy(tr.xn);
                    //evaluate basis in input = [x; u]
                    a(0) = nextAction;
                    arma::vec& tmp = tr.xn;
                    input = arma::join_vert(tmp, a);
                    nextPhi = basis(input);
                    PiPhihat.row(idx) = nextPhi.t();
#ifdef REMOVE_LAST
                }
#endif
                // increment sample counter
                ++idx;
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
            std::cout << "Matrix is not invertible" << std::endl;
            w = arma::pinv(A)*b;
        }

        return w;
    }

    virtual arma::vec run_slow() = 0;

    virtual void computeAandB(const arma::mat& PiPhihat, arma::mat& A, arma::vec& b) = 0;

    virtual ~LSTDQ_()
    {
    }

    e_GreedyApproximate& getPolicy()
    {
        return policy;
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
    using Base::data;
    using Base::Q;
    using Base::policy;
    using Base::gamma;
    using Base::Phihat;
    using Base::Rhat;

public:
    LSTDQ(Dataset<ActionC, DenseState>& data, e_GreedyApproximate& policy,
          Features_<arma::vec>& phi, double gamma)
        : LSTDQ_<ActionC>(data, policy, phi, gamma)
    {
    }

    virtual void computeAandB(const arma::mat& PiPhihat, arma::mat& A, arma::vec& b) override
    {
        A = Phihat.t() * (Phihat - gamma * PiPhihat);
        b = Phihat.t() * Rhat;
    }

    virtual ~LSTDQ()
    {

    }


    arma::vec run_slow() override
    {
        /*** Initialize variables ***/
        int nbEpisodes = data.size();
        //compute the overall number of samples
        int nbSamples = 0;
        for (int k = 0; k < nbEpisodes; ++k)
            nbSamples += data[k].size();

        Features_<arma::vec>& basis = Q.getFeatures();
        int df = basis.rows(); //number of features
        arma::mat A(df, df, arma::fill::zeros);
        arma::vec b(df, arma::fill::zeros);

        arma::vec a(1), input;
        arma::mat phi, nextPhi;
        for (auto episode : data)
        {
            for (auto tr : episode)
            {
                //compute basis in current state-action pair
                a(0) = tr.u;
                arma::vec& tmp = tr.x;
                input = arma::join_vert(tmp, a);
                phi = basis(input);

                //compute basis in next state with action selected by the greedy policy
#ifdef REMOVE_LAST
                if (!tr.xn.isAbsorbing())
                {
#endif
                    typename action_type<ActionC>::type nextAction = policy(tr.xn);
                    a(0) = nextAction;
                    arma::vec& tmp2 = tr.xn;
                    input = arma::join_vert(tmp2, a);
                    nextPhi = basis(input);
#ifdef REMOVE_LAST
                }
                else
                {
                    nextPhi.zeros(phi.n_rows,1);
                }
#endif

                A += phi * (phi - gamma * nextPhi).t();
                b += phi * tr.r[0];
            }
        }
        /*** Solve the system to find w ***/
        arma::vec w;
        int rank = arma::rank(A);
        if (rank == df)
        {
            w = arma::solve(A,b);
        }
        else
        {
            std::cout << "Matrix is not invertible" << std::endl;
            w = arma::pinv(A)*b;
        }

        return w;
    }

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
    using Base::data;
    using Base::Q;
    using Base::policy;
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

    virtual arma::vec run_slow()
    {
        /*** Initialize variables ***/
        int nbEpisodes = data.size();
        //compute the overall number of samples
        int nbSamples = 0;
        for (int k = 0; k < nbEpisodes; ++k)
            nbSamples += data[k].size();

        Features_<arma::vec>& basis = Q.getBasis();
        int df = basis.rows(); //number of features
        arma::mat A(df, df, arma::fill::zeros);
        arma::vec b(df, arma::fill::zeros);

        arma::vec a(1), input;
        arma::mat phi, nextPhi;
        for (auto episode : data)
        {
            for (auto tr : episode)
            {
                //compute basis in current state-action pair
                a(0) = tr.u;
                input = arma::join_vert(tr.x, a);
                phi = basis(input);

                //compute basis in next state with action selected by the greedy policy
                typename action_type<ActionC>::type nextAction = policy(tr.xn);
                a(0) = nextAction;
                input = arma::join_vert(tr.xn, a);
                nextPhi = basis(input);


                A += (phi - gamma * nextPhi) * (phi - gamma * nextPhi).t();
                b += (phi - gamma * nextPhi) * tr.r[0];
            }
        }
        /*** Solve the system to find w ***/
        arma::vec w;
        int rank = arma::rank(A);
        if (rank == df)
        {
            w = arma::solve(A,b);
        }
        else
        {
            std::cout << "Matrix is not invertible" << std::endl;
            w = arma::pinv(A)*b;
        }

        return w;
    }

    virtual ~LSTDQBe()
    {

    }

};

}//end namespace

#endif //LSTDQ_H_
