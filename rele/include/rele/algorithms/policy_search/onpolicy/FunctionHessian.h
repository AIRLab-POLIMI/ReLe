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

#ifndef FUNCTIONHESSIAN_H_
#define FUNCTIONHESSIAN_H_

#include "Basics.h"
#include "Transition.h"
#include "policy_search/onpolicy/PolicyGradientAlgorithm.h"
#include "RewardTransformation.h"


namespace ReLe
{

template<class ActionC, class StateC, class PolicyC>
class HessianFromDataWorker
{
public:

    HessianFromDataWorker(Dataset<ActionC,StateC>& dataset,
                          PolicyC& policy,
                          double gamma, int reward_obj = 0)
        : policy(policy), data(dataset), rewardf(new IndexRT(reward_obj)),
          gamma(gamma), cleanRT(true)
    {
    }

    HessianFromDataWorker(Dataset<ActionC,StateC>& dataset,
                          PolicyC& policy,
                          RewardTransformation& rewardf,
                          double gamma)
        : policy(policy), data(dataset), rewardf(&rewardf),
          gamma(gamma), cleanRT(false)
    {
    }

    virtual ~HessianFromDataWorker()
    {
        if (cleanRT)
        {
            delete rewardf;
        }
    }

    arma::mat ReinforceHessian()
    {
        int dp  = policy.getParametersSize();
        arma::vec sumGradLog(dp), localg;
        arma::mat sumHessLog(dp,dp), localh;
        arma::mat hessian_J(dp, dp, arma::fill::zeros);
        double Rew;

        int nbEpisodes = data.size();
        for (int i = 0; i < nbEpisodes; ++i)
        {
            //core setup
            int nbSteps = data[i].size();


            // *** REINFORCE CORE *** //
            sumGradLog.zeros();
            sumHessLog.zeros();
            double df = 1.0;
            Rew = 0.0;
            // ********************** //

            //iterate the episode
            for (int t = 0; t < nbSteps; ++t)
            {
                Transition<ActionC, StateC>& tr = data[i][t];
                //            std::cout << tr.x << " " << tr.u << " " << tr.xn << " " << tr.r[0] << std::endl;

                // *** REINFORCE CORE *** //
                localg = policy.difflog(tr.x, tr.u);
                localh = policy.diff2log(tr.x, tr.u);
                sumGradLog += localg;
                sumHessLog += localh;
                Rew += df * rewardf->operator ()(tr.r);
                // ********************** //

                df *= gamma;

                if (tr.xn.isAbsorbing())
                {
                    assert(nbSteps == t+1);
                    break;
                }
            }

            // *** REINFORCE CORE *** //
            arma::mat H1 = sumGradLog * sumGradLog.t();
            hessian_J += Rew * (H1 + sumHessLog);
            // ********************** //

        }
        // compute mean values
        hessian_J /= nbEpisodes;

        return hessian_J;
    }

//    arma::mat ReinforceBaseHessian()
//    {
//        int dp  = policy.getParametersSize();
//        int nbEpisodes = data.size();

//        arma::vec sumGradLog(dp), localg;
//        arma::vec gradient_J(dp, arma::fill::zeros);
//        double Rew;

//        arma::vec baseline_J_num(dp, arma::fill::zeros);
//        arma::vec baseline_den(dp, arma::fill::zeros);
//        arma::vec return_J_ObjEp(nbEpisodes);
//        arma::mat sumGradLog_CompEp(dp,nbEpisodes);

//        for (int i = 0; i < nbEpisodes; ++i)
//        {
//            //core setup
//            int nbSteps = data[i].size();


//            // *** REINFORCE CORE *** //
//            sumGradLog.zeros();
//            double df = 1.0;
//            Rew = 0.0;
//            // ********************** //

//            //iterate the episode
//            for (int t = 0; t < nbSteps; ++t)
//            {
//                Transition<ActionC, StateC>& tr = data[i][t];
//                //            std::cout << tr.x << " " << tr.u << " " << tr.xn << " " << tr.r[0] << std::endl;

//                // *** REINFORCE CORE *** //
//                localg = diffLogWorker(tr.x, tr.u, policy);
//                sumGradLog += localg;
//                Rew += df * rewardf->operator ()(tr.r);
//                // ********************** //

//                df *= gamma;

//                if (tr.xn.isAbsorbing())
//                {
//                    assert(nbSteps == t+1);
//                    break;
//                }
//            }

//            // *** REINFORCE BASE CORE *** //

//            // store the basic elements used to compute the gradients

//            return_J_ObjEp(i) = Rew;

//            for (int p = 0; p < dp; ++p)
//            {
//                sumGradLog_CompEp(p,i) = sumGradLog(p);
//            }

//            // compute the baselines
//            for (int p = 0; p < dp; ++p)
//            {
//                baseline_J_num(p) += Rew * sumGradLog(p) * sumGradLog(p);
//                baseline_den(p) += sumGradLog(p) * sumGradLog(p);
//            }

//            // ********************** //

//        }

//        // *** REINFORCE BASE CORE *** //

//        // compute the gradients
//        for (int p = 0; p < dp; ++p)
//        {

//            double baseline_J = 0;
//            if (baseline_den(p) != 0)
//            {
//                baseline_J = baseline_J_num(p) / baseline_den(p);
//            }

//            for (int ep = 0; ep < nbEpisodes; ++ep)
//            {
//                gradient_J[p] += (return_J_ObjEp(ep) - baseline_J) * sumGradLog_CompEp(p,ep);
//            }
//        }

//        // ********************** //

//        // compute mean values
//        gradient_J /= nbEpisodes;

//        return gradient_J;
//    }

    void setPolicy(DifferentiablePolicy<ActionC,StateC>& policy)
    {
        this->policy = policy;
    }

    virtual DifferentiablePolicy<ActionC, StateC>* getPolicy()
    {
        return &policy;
    }

    void setData(Dataset<ActionC,StateC>& dataset)
    {
        data = dataset;
    }

    Dataset<ActionC,StateC>& setData()
    {
        return data;
    }

protected:
    Dataset<ActionC,StateC>& data;
    PolicyC& policy;
    RewardTransformation* rewardf;
    double gamma;
    bool cleanRT;
};

}

#endif //FUNCTIONHESSIAN_H_

