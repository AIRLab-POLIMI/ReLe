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

#ifndef NES_H_
#define NES_H_

#include "policy_search/PGPE/PGPE.h"

namespace ReLe
{

template<class ActionC, class StateC>
class NES: public PGPE<ActionC, StateC>
{

    typedef PGPE<ActionC, StateC> Base;
public:
    NES(DifferentiableDistribution& dist, ParametricPolicy<ActionC, StateC>& policy,
        unsigned int nbEpisodes, unsigned int nbPolicies, double step_length,
        bool baseline = true, int reward_obj = 0)
        : PGPE<ActionC, StateC>(dist, policy, nbEpisodes, nbPolicies, step_length, baseline, reward_obj)
    {
        // create statistic for first iteration
        PGPEIterationStats trace;
        trace.metaParams = dist.getParameters();
        Base::traces.push_back(trace);
    }

    virtual ~NES()
    {}

protected:
    virtual void init()
    {
        int dp = Base::policy.getParametersSize();
        Base::diffObjFunc = arma::vec(dp, arma::fill::zeros);
        b_num = arma::vec(dp, arma::fill::zeros);
        b_den = arma::vec(dp, arma::fill::zeros);
        fisherMtx = arma::mat(dp,dp, arma::fill::zeros);
        Base::history_dlogsist.assign(Base::nbPoliciesToEvalMetap, Base::diffObjFunc);
        Base::history_J = arma::vec(Base::nbPoliciesToEvalMetap, arma::fill::zeros);
    }

    virtual void afterPolicyEstimate()
    {
        //average over episodes
        Base::Jpol /= Base::nbEpisodesToEvalPolicy;
        Base::history_J[Base::polCount] = Base::Jpol;

        //compute gradient log distribution
        const arma::vec& theta = Base::policy.getParameters();
        arma::vec dlogdist = Base::dist.difflog(theta); //\nabla \log D(\theta|\rho)

        //compute baseline
        Base::history_dlogsist[Base::polCount] = dlogdist; //save gradients for late processing
        arma::vec dlogdist2 = (dlogdist % dlogdist);
        b_num += Base::Jpol * dlogdist2;
        b_den += dlogdist2;


        ++(Base::polCount); //until now polCount policies have been analyzed
        Base::epiCount = 0;
        Base::Jpol = 0.0;
    }

    virtual void afterMetaParamsEstimate()
    {

        //compute baseline
        arma::vec baseline = b_num;
        if (Base::useBaseline)
        {
            for (int i = 0, ie = baseline.n_elem; i < ie; ++i)
                if (b_den[i] != 0)
                {
                    baseline[i] /= b_den[i];
                }
                else
                {
                    baseline[i] = 0;
                }
        }
        else
        {
            baseline.zeros();
        }

        Base::diffObjFunc.zeros();
        fisherMtx.zeros();
        //Estimate gradient and Fisher information matrix
        for (int i = 0; i < Base::polCount; ++i)
        {
            Base::diffObjFunc += (Base::history_dlogsist[i] - baseline) * Base::history_J[i];
            fisherMtx += Base::history_dlogsist[i] * (Base::history_dlogsist[i].t());
        }
        Base::diffObjFunc /= Base::polCount;
        fisherMtx /= Base::polCount;

//        arma::mat tmp;
//        int rnk = arma::rank(fisherMtx);
//        if (rnk == fisherMtx.n_rows)
//        {
//            arma::mat H = arma::solve(fisherMtx, diffObjFunc);
//            tmp = diffObjFunc.t() * H;
//        } else {
//            std::cerr << "WARNING: Fisher Matrix is lower rank (rank = " << rnk << ")!!! Should be " << fisherMtx.n_rows << std::endl;
//            arma::mat H = arma::pinv(fisherMtx);
//            tmp = diffObjFunc.t() * (H * diffObjFunc);
//        }

//        double lambda = sqrt(tmp(0,0) / (4 * step_length));
//        lambda = std::max(lambda, 1e-8); // to avoid numerical problems
//        arma::vec nat_grad = arma::solve(fisherMtx, diffObjFunc) / (2 * lambda);

        arma::mat tmp;
        int rnk = arma::rank(fisherMtx);
        if (rnk == fisherMtx.n_rows)
        {
            arma::mat H = arma::solve(fisherMtx, Base::diffObjFunc);
            tmp = H;
        }
        else
        {
            std::cerr << "WARNING: Fisher Matrix is lower rank (rank = " << rnk << ")!!! Should be " << fisherMtx.n_rows << std::endl;
            arma::mat H = arma::pinv(fisherMtx);
            tmp = H * Base::diffObjFunc;
        }

        arma::vec nat_grad = tmp*Base::step_length;


        //--------- save value of distgrad
        int dim = Base::traces.size() - 1;
        Base::traces[dim].metaGradient = nat_grad;
        //---------

        //update meta distribution
        Base::dist.update(nat_grad);


        std::cout << "nat_grad: " << nat_grad.t();
        std::cout << "Parameters:\n" << std::endl;
        std::cout << Base::dist.getParameters() << std::endl;


        //reset counters and gradient
        Base::polCount = 0;
        Base::epiCount = 0;
        Base::runCount++;
        for (int i = 0, ie = Base::diffObjFunc.n_elem; i < ie; ++i)
        {
            //diffObjFunc[i] = 0.0;
            b_num[i] = 0.0;
            b_den[i] = 0.0;
        }

        //---------  create statistic for first iteration
        PGPEIterationStats trace;
        trace.metaParams = Base::dist.getParameters();
        Base::traces.push_back(trace);
        //---------
    }

protected:
    arma::vec b_num, b_den;
    arma::mat fisherMtx;

};


} //end namespace

#endif //NES_H_
