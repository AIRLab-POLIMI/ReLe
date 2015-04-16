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

#ifndef GIRL_H_
#define GIRL_H_

#include "IRLAlgorithm.h"
#include "Policy.h"
#include "Transition.h"
#include "policy_search/onpolicy/PolicyGradientAlgorithm.h"
#include <nlopt.hpp>

namespace ReLe
{

//TODO togliere, questo e' solo temporaneo
template<class ActionC, class StateC>
class IRLParametricReward
{
public:
    virtual double operator()(StateC& s, ActionC& a, StateC& ns) = 0;
    virtual arma::mat diff(StateC& s, ActionC& a, StateC& ns) = 0;

    virtual inline void setParameters(unsigned int n, const double* x)
    {
        assert(weights.n_elem == n);
        for (unsigned int i = 0; i < n; ++i)
            weights[i] = x[i];
    }

    virtual inline void setParameters(arma::vec& params)
    {
        weights = params;
    }

    virtual inline unsigned int getParametersSize()
    {
        return weights.n_elem;
    }

    virtual inline arma::vec getParameters()
    {
        return weights;
    }
protected:
    arma::vec weights;
};

template<class ActionC, class StateC>
class GIRL : public IRLAlgorithm<ActionC, StateC>
{
public:
    enum AlgType {R, RB, G, GB};

    GIRL(Dataset<ActionC,StateC>& dataset,
         DifferentiablePolicy<ActionC,StateC>& policy,
         IRLParametricReward<ActionC, StateC>& rewardf,
         double gamma, AlgType aType)
        : policy(policy), data(dataset), rewardf(rewardf),
          gamma(gamma), maxSteps(0), atype(aType)
    {
    }

    virtual ~GIRL() { }

    virtual void run()
    {
        maxSteps = 0;
        int nbEpisodes = data.size();
        for (int i = 0; i < nbEpisodes; ++i)
        {
            int nbSteps = data[i].size();
            if (maxSteps < nbSteps)
                maxSteps = nbSteps;
        }

        int dpr = rewardf.getParametersSize();
        assert(dpr > 0);

        //setup optimization algorithm
        nlopt::opt optimizator;


        //                optimizator = nlopt::opt(nlopt::algorithm::LD_SLSQP, dpr);
        //                optimizator = nlopt::opt(nlopt::algorithm::AUGLAG, dpr);
        //                optimizator.set_local_optimizer(localoptimizator);
        //        optimizator = nlopt::opt(nlopt::algorithm::LD_MMA, dpr);

        //              optimizator = nlopt::opt(nlopt::algorithm::GN_ORIG_DIRECT, dpr);
        //                      optimizator = nlopt::opt(nlopt::algorithm::GN_ORIG_DIRECT_L, dpr);
        optimizator = nlopt::opt(nlopt::algorithm::LN_COBYLA, dpr);
        optimizator.set_min_objective(GIRL::wrapper, this);
        optimizator.set_xtol_rel(1e-10);
        optimizator.set_ftol_rel(1e-14);
        optimizator.set_maxeval(300*dpr);

        std::vector<double> lowerBounds(dpr, 0.0);
        std::vector<double> upperBounds(dpr, 1.0);
        optimizator.set_lower_bounds(lowerBounds);
        optimizator.set_upper_bounds(upperBounds);
        optimizator.add_equality_constraint(GIRL::OneSumConstraint, NULL, 1e-8);
        //        optimizator.add_inequality_constraint(GIRL::f1, NULL, 1e-16);
        //        optimizator.add_inequality_constraint(GIRL::f2, NULL, 1e-16);


        //optimize dual function
        std::vector<double> parameters(dpr, 0.0);
        double minf;
        if (optimizator.optimize(parameters, minf) < 0)
        {
            printf("nlopt failed!\n");
        }
        else
        {
            printf("found minimum = %0.10g\n", minf);

            arma::vec finalP(dpr);
            for(int i = 0; i < dpr; ++i)
            {
                finalP(i) = parameters[i];
            }
            std::cout << std::endl;

            rewardf.setParameters(finalP);
        }
    }

    virtual arma::vec getWeights()
    {
        return rewardf.getParameters();
    }

    virtual Policy<ActionC, StateC>* getPolicy()
    {
        return &policy;
    }

    void setData(Dataset<ActionC,StateC>& dataset)
    {
        data = dataset;
    }

    arma::vec ReinforceGradient(arma::mat& gGradient)
    {
        int dp  = policy.getParametersSize();
        int dpr = rewardf.getParametersSize();

        gGradient.zeros(dp,dpr);

        arma::vec sumGradLog(dp), localg;
        arma::vec gradient_J(dp, arma::fill::zeros);
        double Rew;
        arma::mat dRew(1,dpr);

        int nbEpisodes = data.size();
        for (int i = 0; i < nbEpisodes; ++i)
        {
            //core setup
            int nbSteps = data[i].size();


            // *** REINFORCE CORE *** //
            sumGradLog.zeros();
            double df = 1.0;
            Rew = 0.0;
            dRew.zeros();
            // ********************** //

            //iterate the episode
            for (int t = 0; t < nbSteps; ++t)
            {
                Transition<ActionC, StateC>& tr = data[i][t];
                //            std::cout << tr.x << " " << tr.u << " " << tr.xn << " " << tr.r[0] << std::endl;

                // *** REINFORCE CORE *** //
                localg = diffLogWorker(tr.x, tr.u, policy);
                sumGradLog += localg;
//                std::cout << tr.r[0] << " " << tr.r[1] << std::endl;
                Rew += df * rewardf(tr.x, tr.u, tr.xn);
                dRew += df * rewardf.diff(tr.x, tr.u, tr.xn);
                // ********************** //

                df *= gamma;

                if (tr.xn.isAbsorbing())
                {
                    assert(nbSteps == t+1);
                    break;
                }
            }

            // *** REINFORCE CORE *** //
            for (int p = 0; p < dp; ++p)
            {
                gradient_J[p] += Rew * sumGradLog(p);
                for (int rp = 0; rp < dpr; ++rp)
                {
                    gGradient(p,rp) += sumGradLog(p) * dRew(0,rp);
                }
            }
            // ********************** //

        }
        // compute mean values
        gradient_J /= nbEpisodes;
        gGradient  /= nbEpisodes;

        return gradient_J;
    }

    arma::vec ReinforceBaseGradient(arma::mat& gGradient)
    {
        int dp  = policy.getParametersSize();
        int dpr = rewardf.getParametersSize();
        int nbEpisodes = data.size();

        gGradient.zeros(dp,dpr);

        arma::vec sumGradLog(dp), localg;
        arma::vec gradient_J(dp, arma::fill::zeros);
        double Rew;
        arma::mat dRew(1,dpr);

        arma::vec baseline_J_num(dp, arma::fill::zeros);
        arma::vec baseline_den(dp, arma::fill::zeros);
        arma::vec return_J_ObjEp(nbEpisodes);
        arma::mat sumGradLog_CompEp(dp,nbEpisodes);

        arma::mat baseline_R_num(dp, dpr, arma::fill::zeros);
        std::vector<arma::mat> return_R_ObjEp(nbEpisodes, arma::mat());

        for (int i = 0; i < nbEpisodes; ++i)
        {
            //core setup
            int nbSteps = data[i].size();


            // *** REINFORCE CORE *** //
            sumGradLog.zeros();
            double df = 1.0;
            Rew = 0.0;
            dRew.zeros();
            // ********************** //

            //iterate the episode
            for (int t = 0; t < nbSteps; ++t)
            {
                Transition<ActionC, StateC>& tr = data[i][t];
                //            std::cout << tr.x << " " << tr.u << " " << tr.xn << " " << tr.r[0] << std::endl;

                // *** REINFORCE CORE *** //
                localg = diffLogWorker(tr.x, tr.u, policy);
                sumGradLog += localg;
                Rew += df * rewardf(tr.x, tr.u, tr.xn);
                dRew += df * rewardf.diff(tr.x, tr.u, tr.xn);
                // ********************** //

                df *= gamma;

                if (tr.xn.isAbsorbing())
                {
                    assert(nbSteps == t+1);
                    break;
                }
            }

            // *** REINFORCE BASE CORE *** //

            // store the basic elements used to compute the gradients

            return_J_ObjEp(i) = Rew;
            return_R_ObjEp[i] = dRew;

            for (int p = 0; p < dp; ++p)
            {
                sumGradLog_CompEp(p,i) = sumGradLog(p);
            }

            // compute the baselines
            for (int p = 0; p < dp; ++p)
            {
                baseline_J_num(p) += Rew * sumGradLog(p) * sumGradLog(p);
                baseline_den(p) += sumGradLog(p) * sumGradLog(p);
                for (int rp = 0; rp < dpr; ++rp)
                {
                    baseline_R_num(p,rp) += sumGradLog(p) * sumGradLog(p) * dRew(0,rp);
                }
            }

            // ********************** //

        }

        // *** REINFORCE BASE CORE *** //

        // compute the gradients
        for (int p = 0; p < dp; ++p)
        {

            double baseline_J = 0;
            if (baseline_den(p) != 0)
            {
                baseline_J = baseline_J_num(p) / baseline_den(p);
            }

            for (int ep = 0; ep < nbEpisodes; ++ep)
            {
                gradient_J[p] += (return_J_ObjEp(ep) - baseline_J) * sumGradLog_CompEp(p,ep);

                for (int rp = 0; rp < dpr; ++rp)
                {
                    double basel = baseline_den(p) != 0 ? baseline_R_num(p,rp) / baseline_den(p) : 0.0;
                    gGradient(p,rp) += (return_R_ObjEp[ep](0,rp) - basel) * sumGradLog_CompEp(p,ep);
                }
            }
        }

        // ********************** //

        // compute mean values
        gradient_J /= nbEpisodes;
        gGradient  /= nbEpisodes;

        return gradient_J;
    }

    arma::vec GpomdpGradient(arma::mat& gGradient)
    {
        int dp  = policy.getParametersSize();
        int dpr = rewardf.getParametersSize();

        gGradient.zeros(dp,dpr);

        arma::vec sumGradLog(dp), localg;
        arma::vec gradient_J(dp, arma::fill::zeros);
        double Rew;
        arma::mat dRew(1,dpr);

        int nbEpisodes = data.size();
        for (int i = 0; i < nbEpisodes; ++i)
        {
            //core setup
            int nbSteps = data[i].size();


            // *** GPOMDP CORE *** //
            sumGradLog.zeros();
            double df = 1.0;
            Rew = 0.0;
            dRew.zeros();
            // ********************** //

            //iterate the episode
            for (int t = 0; t < nbSteps; ++t)
            {
                Transition<ActionC, StateC>& tr = data[i][t];
                //            std::cout << tr.x << " " << tr.u << " " << tr.xn << " " << tr.r[0] << std::endl;

                // *** GPOMDP CORE *** //
                localg = diffLogWorker(tr.x, tr.u, policy);
                sumGradLog += localg;
                Rew += df * rewardf(tr.x, tr.u, tr.xn);
                dRew += df * rewardf.diff(tr.x, tr.u, tr.xn);

                // compute the gradients
                Rew += df * rewardf(tr.x, tr.u, tr.xn);
                dRew += df * rewardf.diff(tr.x, tr.u, tr.xn);

                for (int p = 0; p < dp; ++p)
                {
                    gradient_J[p] += df * rewardf(tr.x, tr.u, tr.xn) * sumGradLog(p);
                    for (int rp = 0; rp < dpr; ++rp)
                    {
                        gGradient(p,rp) += sumGradLog(p) * dRew(0,rp);
                    }
                }
                // ********************** //

                df *= gamma;

                if (tr.xn.isAbsorbing())
                {
                    assert(nbSteps == t+1);
                    break;
                }
            }

        }
        // compute mean values
        gradient_J /= nbEpisodes;
        gGradient  /= nbEpisodes;

        return gradient_J;
    }

    arma::vec GpomdpBaseGradient(arma::mat& gGradient)
    {
        int dp  = policy.getParametersSize();
        int dpr = rewardf.getParametersSize();
        int nbEpisodes = data.size();

        gGradient.zeros(dp,dpr);

        arma::vec sumGradLog(dp), localg;
        arma::vec gradient_J(dp, arma::fill::zeros);
        double Rew;
        arma::mat dRew(1,dpr);


        arma::mat baseline_J_num(dp, maxSteps, arma::fill::zeros);
        std::vector<arma::mat> baseline_R_num(maxSteps, arma::mat(dp,dpr, arma::fill::zeros));
        arma::mat baseline_den(dp, maxSteps, arma::fill::zeros);
        arma::mat reward_J_ObjEpStep(nbEpisodes, maxSteps);
        std::vector<std::vector<arma::mat>> reward_R_ObjEpStep(nbEpisodes,
                                         std::vector<arma::mat>(maxSteps,arma::mat()));
        arma::cube sumGradLog_CompEpStep(dp,nbEpisodes, maxSteps);
        arma::vec  maxsteps_Ep(nbEpisodes);

        for (int ep = 0; ep < nbEpisodes; ++ep)
        {
            //core setup
            int nbSteps = data[ep].size();


            // *** GPOMDP CORE *** //
            sumGradLog.zeros();
            double df = 1.0;
            Rew = 0.0;
            dRew.zeros();
            // ********************** //

            //iterate the episode
            for (int t = 0; t < nbSteps; ++t)
            {
                Transition<ActionC, StateC>& tr = data[ep][t];
                //            std::cout << tr.x << " " << tr.u << " " << tr.xn << " " << tr.r[0] << std::endl;

                // *** GPOMDP CORE *** //
                localg = diffLogWorker(tr.x, tr.u, policy);
                sumGradLog += localg;

                // store the basic elements used to compute the gradients
                double creward = rewardf(tr.x, tr.u, tr.xn);
                arma::mat cdreward = rewardf.diff(tr.x, tr.u, tr.xn);
                Rew += df * creward;
                dRew += df * cdreward;
                reward_J_ObjEpStep(ep,t) = df * creward;
                reward_R_ObjEpStep[ep][t] = df * cdreward;


                for (int p = 0; p < dp; ++p)
                {
                    sumGradLog_CompEpStep(p,ep,t) = sumGradLog(p);
                }

                // compute the baselines
                for (int p = 0; p < dp; ++p)
                {
                    baseline_J_num(p,t) += df * creward * sumGradLog(p) * sumGradLog(p);

                    for (int rp = 0; rp < dpr; ++rp)
                    {
                        baseline_R_num[t](p,rp) += df * cdreward(0,rp) * sumGradLog(p) * sumGradLog(p);
                    }
                }

                for (int p = 0; p < dp; ++p)
                {
                    baseline_den(p,t) += sumGradLog(p) * sumGradLog(p);
                }
                // ********************** //

                df *= gamma;

                if (tr.xn.isAbsorbing())
                {
                    assert(nbSteps == t+1);
                    break;
                }
            }

            // store the actual length of the current episode (<= maxsteps)
            maxsteps_Ep(ep) = nbSteps;

        }

        // *** GPOMDP BASE CORE *** //

        // compute the gradients
        for (int p = 0; p < dp; ++p)
        {
            for (int ep = 0; ep < nbEpisodes; ++ep)
            {
                for (int t = 0; t < maxsteps_Ep(ep); ++t)
                {

                    double baseline_J = 0;
                    if (baseline_den(p,t) != 0)
                    {
                        baseline_J = baseline_J_num(p,t) / baseline_den(p,t);
                    }

                    gradient_J[p] += (reward_J_ObjEpStep(ep,t) - baseline_J) * sumGradLog_CompEpStep(p,ep,t);

                    arma::mat& tmp = reward_R_ObjEpStep[ep][t];
                    for (int rp = 0; rp < dpr; ++rp)
                    {
                        double basel = baseline_den(p) != 0 ? baseline_R_num[t](p,rp) / baseline_den(p) : 0.0;
                        gGradient(p,rp) += (tmp(0,rp) - basel) * sumGradLog_CompEpStep(p,ep,t);
                    }
                }
            }
        }
        // ************************ //

        // compute mean values
        gradient_J /= nbEpisodes;
        gGradient  /= nbEpisodes;

        return gradient_J;
    }


    double objFunction(unsigned int n, const double* x, double* grad)
    {
        arma::vec gradient;
        arma::mat dGradient;
        rewardf.setParameters(n, x);
        if (atype == AlgType::R)
        {
//            std::cout << "GIRL REINFORCE" << std::endl;
            gradient = ReinforceGradient(dGradient);
        }
        else if (atype == AlgType::RB)
        {
//            std::cout << "GIRL REINFORCE BASE" << std::endl;
            gradient = ReinforceBaseGradient(dGradient);
        }
        else if (atype == AlgType::G)
        {
//            std::cout << "GIRL GPOMDP" << std::endl;
            gradient = GpomdpGradient(dGradient);
        }
        else if (atype == AlgType::GB)
        {
//            std::cout << "GIRL GPOMDP BASE" << std::endl;
            gradient = GpomdpBaseGradient(dGradient);
        }

        //        std::cerr << gradient.t();
        //        std::cerr << dGradient;

        if (grad != nullptr)
        {
            arma::vec g = dGradient.t() * gradient;
            //            std::cerr << g.t();
            for (int i = 0; i < g.n_elem; ++i)
            {
                grad[i] = g[i];
            }
        }

        double norm22 = arma::norm(gradient,2);
        double f = 0.5 * norm22 * norm22;
        //        std::cerr << f << std::endl;
        return f;

    }

    static double OneSumConstraint(unsigned int n, const double *x, double *grad, void *data)
    {
        grad = nullptr;
        double val = -1.0;
        for (unsigned int i = 0; i < n; ++i)
        {
            val += x[i];
        }
        return val;
    }

    static double f1(unsigned int n, const double *x, double *grad, void *data)
    {
        grad = nullptr;
        double val = -1.0;
        for (unsigned int i = 0; i < n; ++i)
        {
            val += x[i];
            if (grad != nullptr)
                grad[i] = 1;
        }
        return val;
    }
    static double f2(unsigned int n, const double *x, double *grad, void *data)
    {
        double val = 1.0;
        for (unsigned int i = 0; i < n; ++i)
        {
            val += -x[i];
            if (grad != nullptr)
                grad[i] = -1;
        }
        return val;
    }

    static double wrapper(unsigned int n, const double* x, double* grad,
                          void* o)
    {
        return reinterpret_cast<GIRL*>(o)->objFunction(n, x, grad);
    }

protected:
    Dataset<ActionC,StateC>& data;
    DifferentiablePolicy<ActionC,StateC>& policy;
    IRLParametricReward<ActionC, StateC>& rewardf;
    double gamma;
    unsigned int maxSteps;
    AlgType atype;
};


}


#endif /* GIRL_H_ */
