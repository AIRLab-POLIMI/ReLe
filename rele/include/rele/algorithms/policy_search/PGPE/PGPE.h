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

#ifndef PGPE_H_
#define PGPE_H_

#include "rele/statistics/Distribution.h"
#include "rele/algorithms/policy_search/BlackBoxAlgorithm.h"
#include "rele/policy/Policy.h"
#include "rele/core/Basics.h"
#include <cassert>
#include <iomanip>

#include "PGPEOutputData.h"

namespace ReLe
{

template<class ActionC, class StateC>
class PGPE: public GradientBlackBoxAlgorithm<ActionC, StateC, PGPEIterationStats>
{
    typedef GradientBlackBoxAlgorithm<ActionC, StateC, PGPEIterationStats> Base;
public:
    PGPE(DifferentiableDistribution& dist, ParametricPolicy<ActionC, StateC>& policy,
         unsigned int nbEpisodes, unsigned int nbPolicies, GradientStep& step_length,
         bool baseline = true, int reward_obj = 0)
        : GradientBlackBoxAlgorithm<ActionC, StateC, PGPEIterationStats>
        (dist, policy, nbEpisodes, nbPolicies, step_length, baseline, reward_obj),
        useDirection(false)
    {
    }

    PGPE(DifferentiableDistribution& dist, ParametricPolicy<ActionC, StateC>& policy,
         unsigned int nbEpisodes, unsigned int nbPolicies, GradientStep& step_length,
         RewardTransformation& reward_tr,
         bool baseline = true)
        : GradientBlackBoxAlgorithm<ActionC, StateC, PGPEIterationStats>
        (dist, policy, nbEpisodes, nbPolicies, step_length, reward_tr, baseline),
        useDirection(false)
    {
    }


    virtual ~PGPE() {}

    inline virtual void setNormalization(bool flag)
    {
        this->useDirection = flag;
    }

    inline virtual bool isNormalized()
    {
        return this->useDirection;
    }

protected:
    virtual void init() override
    {
        int dp = Base::dist.getParametersSize();
        Base::diffObjFunc = arma::vec(dp, arma::fill::zeros);
        Base::history_dlogsist.assign(Base::nbPoliciesToEvalMetap, Base::diffObjFunc);
        Base::history_J = arma::vec(Base::nbPoliciesToEvalMetap, arma::fill::zeros);

        bm_num = arma::vec(dp, arma::fill::zeros);
        bm_den = arma::vec(dp, arma::fill::zeros);

    }

    virtual void afterPolicyEstimate() override
    {
        //average over episodes
        Base::Jpol /= Base::nbEpisodesToEvalPolicy;
        Base::history_J[Base::polCount] = Base::Jpol;

        //compute gradient log distribution
        const arma::vec& theta = Base::policy.getParameters();
        arma::vec dlogdist = Base::dist.difflog(theta); //\nabla \log D(\theta|\rho)

        //compute baseline
        Base::history_dlogsist[Base::polCount] = dlogdist; //save gradients for late processing

        //multi-baseline
        arma::vec dlogdist2 = (dlogdist % dlogdist);
        bm_num += Base::Jpol * dlogdist2;
        bm_den += dlogdist2;

        //--------- save value of distgrad
        Base::currentItStats->individuals[Base::polCount].diffLogDistr = dlogdist;
        //---------
    }

    virtual void afterMetaParamsEstimate() override
    {

        //compute baseline
        arma::vec baseline = bm_num;
        if (Base::useBaseline)
        {
            for (int i = 0, ie = baseline.n_elem; i < ie; ++i)
                if (bm_den[i] != 0)
                {
                    baseline[i] /= bm_den[i];
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
            Base::diffObjFunc += (Base::history_dlogsist[i]) % (Base::history_J[i] - baseline);

        Base::diffObjFunc /= Base::polCount;


        if (useDirection)
            Base::diffObjFunc = arma::normalise(Base::diffObjFunc);
        //---

        //--------- save value of distgrad
        Base::currentItStats->metaGradient = Base::diffObjFunc;
        //---------


        Base::diffObjFunc = Base::stepLengthRule(Base::diffObjFunc);

        //update meta distribution
        Base::dist.update(Base::diffObjFunc);


        bm_num.zeros();
        bm_den.zeros();

    }

private:

    arma::vec bm_num, bm_den;
    bool useDirection;
};

} //end namespace

#endif //PGPE_H_
