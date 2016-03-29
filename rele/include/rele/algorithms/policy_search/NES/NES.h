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

#include "rele/algorithms/policy_search/PGPE/PGPE.h"
#include "rele/algorithms/policy_search/NES/NESOutputData.h"
#include "rele/statistics/DifferentiableNormals.h"
#include "rele/utils/ArmadilloPDFs.h"

namespace ReLe
{

template<class ActionC, class StateC>
class NES: public GradientBlackBoxAlgorithm<ActionC, StateC, NESIterationStats>
{

    typedef GradientBlackBoxAlgorithm<ActionC, StateC, NESIterationStats> Base;
public:
    NES(DifferentiableDistribution& dist, ParametricPolicy<ActionC, StateC>& policy,
        unsigned int nbEpisodes, unsigned int nbPolicies, GradientStep& step_length,
        bool baseline = true, int reward_obj = 0)
        : GradientBlackBoxAlgorithm<ActionC, StateC, NESIterationStats>
        (dist, policy, nbEpisodes, nbPolicies, step_length, baseline, reward_obj)
    {
        if (dist.getDistributionName().compare("ParametricCholeskyNormal") == 0)
        {
            std::cout << "=================================\nWE SERIOUSLY SUGGEST TO USE ENES WITH CHOLESKY DISTRIBUTION!!\n=================================" << std::endl;
        }
    }

    NES(DifferentiableDistribution& dist, ParametricPolicy<ActionC, StateC>& policy,
        unsigned int nbEpisodes, unsigned int nbPolicies, GradientStep& step_length,
        RewardTransformation& reward_tr,
        bool baseline = true)
        : GradientBlackBoxAlgorithm<ActionC, StateC, NESIterationStats>
        (dist, policy, nbEpisodes, nbPolicies, step_length, reward_tr, baseline)
    {
        if (dist.getDistributionName().compare("ParametricCholeskyNormal") == 0)
        {
            std::cout << "=================================\nWE SERIOUSLY SUGGEST TO USE ENES WITH CHOLESKY DISTRIBUTION!!\n=================================" << std::endl;
        }
    }

    virtual ~NES()
    {
    }

protected:
    virtual void init() override
    {
        int dp = Base::dist.getParametersSize();
        Base::diffObjFunc = arma::vec(dp, arma::fill::zeros);
        b_num = arma::vec(dp, arma::fill::zeros);
        b_den = arma::vec(dp, arma::fill::zeros);
        fisherMtx = arma::mat(dp,dp, arma::fill::zeros);
        Base::history_dlogsist.assign(Base::nbPoliciesToEvalMetap, Base::diffObjFunc);
        Base::history_J = arma::vec(Base::nbPoliciesToEvalMetap, arma::fill::zeros);
    }

    virtual void afterPolicyEstimate() override
    {
        //average over episodes
        Base::Jpol /= Base::nbEpisodesToEvalPolicy;
        Base::history_J[Base::polCount] = Base::Jpol;

        //compute gradient log distribution
        const arma::vec& theta = Base::policy.getParameters();
        arma::vec dlogdist = Base::dist.difflog(theta); //\nabla \log D(\theta|\rho)
        //--------- save value of distgrad
        Base::currentItStats->individuals[Base::polCount].diffLogDistr = dlogdist;
        //---------

        //compute baseline
        Base::history_dlogsist[Base::polCount] = dlogdist; //save gradients for late processing
        arma::vec dlogdist2 = (dlogdist % dlogdist);
        b_num += Base::Jpol * dlogdist2;
        b_den += dlogdist2;
    }

    virtual void afterMetaParamsEstimate() override
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
            //            Base::diffObjFunc += (Base::history_dlogsist[i] - baseline) * Base::history_J[i];
            Base::diffObjFunc += (Base::history_dlogsist[i]) % (Base::history_J[i]- baseline);
            fisherMtx += Base::history_dlogsist[i] * (Base::history_dlogsist[i].t());
        }
        Base::diffObjFunc /= Base::polCount;
        fisherMtx /= Base::polCount;


        //--- Compute learning step

        arma::mat tmp;
        arma::vec nat_grad;
        int rnk = arma::rank(fisherMtx);
        if (rnk == fisherMtx.n_rows)
        {
            arma::mat H = arma::solve(fisherMtx, Base::diffObjFunc);
            nat_grad = arma::solve(fisherMtx, Base::diffObjFunc);
        }
        else
        {
            std::cerr << "WARNING: Fisher Matrix is lower rank (rank = " << rnk << ")!!! Should be " << fisherMtx.n_rows << std::endl;
            arma::mat H = arma::pinv(fisherMtx);
            nat_grad = H * Base::diffObjFunc;
        }

        //--------- save value of distgrad
        Base::currentItStats->metaGradient = nat_grad;
        Base::currentItStats->fisherMtx = fisherMtx;
        //---------

        //update meta distribution
        arma::vec delta = Base::stepLengthRule(Base::diffObjFunc, nat_grad);
        Base::dist.update(delta);

        b_num.zeros();
        b_den.zeros();

    }

protected:
    arma::vec b_num, b_den;
    arma::mat fisherMtx;

};


/**
 * Exact NES (NES with closed-form FIM)
 */
template<class ActionC, class StateC>
class eNES: public GradientBlackBoxAlgorithm<ActionC, StateC, NESIterationStats>
{
    typedef GradientBlackBoxAlgorithm<ActionC, StateC, NESIterationStats> Base;

public:
    eNES(DifferentiableDistribution& dist, ParametricPolicy<ActionC, StateC>& policy,
         unsigned int nbEpisodes, unsigned int nbPolicies, GradientStep& step_length,
         bool baseline = true, int reward_obj = 0)
        : GradientBlackBoxAlgorithm<ActionC, StateC, NESIterationStats>
        (dist, policy, nbEpisodes, nbPolicies, step_length, baseline, reward_obj),
        distFI(dynamic_cast<FisherInterface&>(dist))
    {
    }

    eNES(DifferentiableDistribution& dist, ParametricPolicy<ActionC, StateC>& policy,
         unsigned int nbEpisodes, unsigned int nbPolicies, GradientStep& step_length,
         RewardTransformation& reward_tr,
         bool baseline = true)
        : GradientBlackBoxAlgorithm<ActionC, StateC, NESIterationStats>
        (dist, policy, nbEpisodes, nbPolicies, step_length, reward_tr, baseline),
        distFI(dynamic_cast<FisherInterface&>(dist))
    {
    }

    virtual ~eNES()
    {
    }

protected:
    virtual void init() override
    {
        int dp = Base::dist.getParametersSize();
        Base::diffObjFunc = arma::vec(dp, arma::fill::zeros);
        b_num = arma::vec(dp, arma::fill::zeros);
        b_den = arma::vec(dp, arma::fill::zeros);
        Base::history_dlogsist.assign(Base::nbPoliciesToEvalMetap, Base::diffObjFunc);
        Base::history_J = arma::vec(Base::nbPoliciesToEvalMetap, arma::fill::zeros);
    }

    virtual void afterPolicyEstimate() override
    {
        //average over episodes
        Base::Jpol /= Base::nbEpisodesToEvalPolicy;
        Base::history_J[Base::polCount] = Base::Jpol;

        //compute gradient log distribution
        const arma::vec& theta = Base::policy.getParameters();
        arma::vec dlogdist = Base::dist.difflog(theta); //\nabla \log D(\theta|\rho)
        //--------- save value of distgrad
        Base::currentItStats->individuals[Base::polCount].diffLogDistr = dlogdist;
        //---------

        //compute baseline
        Base::history_dlogsist[Base::polCount] = dlogdist; //save gradients for late processing
        arma::vec dlogdist2 = (dlogdist % dlogdist);
        b_num += Base::Jpol * dlogdist2;
        b_den += dlogdist2;
    }

    virtual void afterMetaParamsEstimate() override
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
        //Estimate gradient and Fisher information matrix
        for (int i = 0; i < Base::polCount; ++i)
        {
            //            Base::diffObjFunc += (Base::history_dlogsist[i] - baseline) * Base::history_J[i];
            Base::diffObjFunc += (Base::history_dlogsist[i]) % (Base::history_J[i] - baseline);
        }
        Base::diffObjFunc /= Base::polCount;


        //--- Compute learning step

        arma::vec delta;
        arma::vec nat_grad;
        arma::sp_mat invFisherMtx = distFI.inverseFIM();
        if (invFisherMtx.n_elem == 0)
        {
            arma::sp_mat spFisherMtx = distFI.FIM();
            arma::mat fisherMtx(spFisherMtx);
            int rnk = arma::rank(fisherMtx);
            if (rnk == fisherMtx.n_rows)
            {
                nat_grad = arma::solve(fisherMtx, Base::diffObjFunc);
            }
            else
            {
                std::cerr << "WARNING: Fisher Matrix is lower rank (rank = " << rnk << ")!!! Should be " << fisherMtx.n_rows << std::endl;
                arma::mat H = arma::pinv(fisherMtx);
                nat_grad = H * Base::diffObjFunc;
            }

            delta = Base::stepLengthRule(Base::diffObjFunc, nat_grad);
        }
        else
        {
            nat_grad = invFisherMtx*Base::diffObjFunc;
            delta = Base::stepLengthRule(Base::diffObjFunc, nat_grad);
        }

        //        std::cout << nat_grad.t();
        if (std::isnan(nat_grad(0)) || std::isnan(delta(0)))
        {
            std::cerr << "Something has gone wrong in eNES" << std::endl;
            abort();
        }

        //--------- save value of distgrad
        Base::currentItStats->metaGradient = nat_grad;
        Base::currentItStats->fisherMtx    = invFisherMtx;
        //---------

        //update meta distribution
        Base::dist.update(delta);

        b_num.zeros();
        b_den.zeros();
    }

protected:
    arma::vec b_num, b_den;
    FisherInterface& distFI;
};

#if 0
/**
 * Exponential NES
 */
template<class ActionC, class StateC>
class xNES: public BlackBoxAlgorithm<ActionC, StateC, ParametricCholeskyNormal, xNESIterationStats>
{
    typedef BlackBoxAlgorithm<ActionC, StateC, ParametricCholeskyNormal, xNESIterationStats> Base;

public:
    xNES(ParametricCholeskyNormal& dist, ParametricPolicy<ActionC, StateC>& policy,
         unsigned int nbEpisodes, unsigned int nbPolicies,
         double step_mu, double step_sigma, double step_b,
         int reward_obj = 0)
        : BlackBoxAlgorithm<ActionC, StateC, ParametricCholeskyNormal, xNESIterationStats>
        (dist, policy, nbEpisodes, nbPolicies, 0.0, false, reward_obj),
        stepmu(step_mu), stepsigma(step_sigma), stepb(step_b)
    {

        arma::mat A = dist.getCholeskyDec();
        double det = arma::det(A);
        double exponent = 1.0 / A.n_rows;
//        std::cout << A << std::endl;
//        std::cout << exponent << std::endl;
        sigma = std::pow(fabs(det), exponent);
//        std::cout << sigma << std::endl;
        B = A/sigma;
        mu = dist.getMean();

//        std::cout << mu.t() << std::endl;
    }

    virtual ~xNES()
    {
    }

protected:
    virtual void initEpisode(const StateC& state, ActionC& action)
    {
        Base::df  = 1.0;    //reset discount factor
        Base::Jep = 0.0;    //reset J of current episode

        if (Base::polCount == 0 && Base::epiCount == 0)
        {
            Base::currentItStats = new xNESIterationStats(Base::nbPoliciesToEvalMetap,
                    Base::policy.getParametersSize(),
                    Base::nbEpisodesToEvalPolicy);
            Base::currentItStats->metaParams = Base::dist.getParameters();
        }

        if (Base::epiCount == 0)
        {
            //a new policy is considered
            Base::Jpol = 0.0;

            //obtain new parameters
            arma::vec ones = arma::ones<arma::vec>(B.n_rows);
            arma::mat I(B.n_rows,B.n_rows, arma::fill::eye);
            arma::vec zi = mvnrandFast(ones,I);
//            std::cout << zi.t();
            arma::vec xi = mu + sigma*B*zi;

            z_individuals[Base::polCount] = zi;
            x_individuals[Base::polCount] = xi;

            //set to policy
            Base::policy.setParameters(xi);

            //--- create new policy individual
            //            currentItStats->individuals.push_back(
            //                PGPEPolicyIndividual(new_params, nbEpisodesToEvalPolicy));
            Base::currentItStats->individuals[Base::polCount].Pparams = xi;
            //---
        }
        Base::sampleAction(state, action);
    }

    virtual void init()
    {
        int dp = Base::dist.getParametersSize();
        Base::diffObjFunc = arma::vec(dp, arma::fill::zeros);
        Base::history_dlogsist.assign(Base::nbPoliciesToEvalMetap, Base::diffObjFunc);
        Base::history_J = arma::vec(Base::nbPoliciesToEvalMetap, arma::fill::zeros);
        z_individuals.assign(Base::nbPoliciesToEvalMetap, arma::vec(Base::policy.getParametersSize()));
        x_individuals.assign(Base::nbPoliciesToEvalMetap, arma::vec(Base::policy.getParametersSize()));
        indicies.assign(Base::nbPoliciesToEvalMetap,0);
    }

    virtual void afterPolicyEstimate()
    {
        //average over episodes
        Base::Jpol /= Base::nbEpisodesToEvalPolicy;
        Base::history_J[Base::polCount] = Base::Jpol;
    }

    virtual void afterMetaParamsEstimate()
    {

        double den_ui = 0.0;
        for (unsigned int i = 0; i < Base::nbPoliciesToEvalMetap; ++i)
        {
            indicies[i] = i;
            //--- compute den of wi
            double A = log(Base::nbPoliciesToEvalMetap/2.0 + 1.0);
            double B = log(1.0*(i+1));
            double C = std::max(0.0, A-B);
            den_ui += C;
            //---
        }
        for (unsigned int i = 0; i < Base::nbPoliciesToEvalMetap - 1; ++i)
        {
            for (unsigned int j = i+1; j < Base::nbPoliciesToEvalMetap; ++j)
            {
                if (Base::history_J[indicies[i]] < Base::history_J[indicies[j]])
                {
                    unsigned int tmp = indicies[i];
                    indicies[i] = indicies[j];
                    indicies[j] = tmp;
                }
            }
        }

        unsigned int d = Base::policy.getParametersSize();
        arma::mat I(d, d, arma::fill::eye), GM(d,d,arma::fill::zeros);
        arma::vec Gdelta(mu.n_elem, arma::fill::zeros);

        double ninv = 1.0 / Base::nbPoliciesToEvalMetap;

        for (unsigned int i = 0; i < Base::nbPoliciesToEvalMetap; ++i)
        {
            //--- compute weight ui
            double A = log(Base::nbPoliciesToEvalMetap/2.0 + 1.0);
            double B = log(1.0*(i+1));
            double C = std::max(0.0, A-B);
            double ui = C/den_ui - ninv;
//            ui = Base::history_J[indicies[i]];
            //---

            unsigned int idx = indicies[i];
            arma::vec& zi = z_individuals[idx];
            Gdelta += ui * zi;
            GM += ui * (zi*zi.t() - I);
        }

        double Gsigma = arma::trace(GM) / d;
        arma::mat GB = GM - Gsigma * I;

        mu += stepmu * sigma * B * Gdelta;
        sigma = sigma * std::exp(stepsigma*Gsigma/2.0);
        B = B * arma::expmat(stepb*GB/2.0);

//        std::cout << mu << std::endl;
//        std::cout << B << std::endl;

        arma::mat tmp = arma::trimatu(arma::ones(d, d));
        arma::vec new_val =  arma::join_vert(mu, sigma*B.elem( arma::find(tmp == 1.0) ));

//        std::cout << new_val.t();
        //update meta distribution
        Base::dist.setParameters(new_val);


        // std::cout << "nat_grad: " << nat_grad.t();
        // std::cout << "Parameters:\n" << std::endl;
        // std::cout << Base::dist.getParameters() << std::endl;

    }

protected:
    arma::mat B;
    arma::vec mu;
    std::vector<arma::vec> z_individuals, x_individuals;
    std::vector<int> indicies;
    double sigma;
    double stepmu, stepsigma, stepb;
};

#endif

} //end namespace

#endif //NES_H_
