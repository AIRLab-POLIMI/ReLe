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

#include "Distribution.h"
#include "policy_search/BBO.h"
#include "Policy.h"
#include "Basics.h"
#include <cassert>
#include <iomanip>

#include "PGPEOutputData.h"

namespace ReLe
{

template<class ActionC, class StateC>
class PGPE: public GradientBlackBoxAlgorithm<ActionC, StateC, DifferentiableDistribution, PGPEIterationStats>
{
    typedef GradientBlackBoxAlgorithm<ActionC, StateC, DifferentiableDistribution, PGPEIterationStats> Base;
public:
    PGPE(DifferentiableDistribution& dist, ParametricPolicy<ActionC, StateC>& policy,
         unsigned int nbEpisodes, unsigned int nbPolicies, double step_length,
         bool baseline = true, int reward_obj = 0)
        : GradientBlackBoxAlgorithm<ActionC, StateC, DifferentiableDistribution, PGPEIterationStats>
        (dist, policy, nbEpisodes, nbPolicies, step_length, baseline, reward_obj),
        b_num(0.0), b_den(0.0), useDirection(false)
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
    virtual void init()
    {
        int dp = Base::dist.getParametersSize();
        Base::diffObjFunc = arma::vec(dp, arma::fill::zeros);
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
        double norm2G2 = arma::norm(dlogdist,2);
        norm2G2 *= norm2G2;
        Base::history_dlogsist[Base::polCount] = dlogdist; //save gradients for late processing
        b_num += Base::Jpol * norm2G2;
        b_den += norm2G2;


        //--------- save value of distgrad
        Base::currentItStats->individuals[Base::polCount].diffLogDistr = dlogdist;
        //---------
    }

    virtual void afterMetaParamsEstimate()
    {

        //compute baseline
        double baseline = (b_den != 0 && Base::useBaseline) ? b_num/b_den : 0.0;

        Base::diffObjFunc.zeros();
        //Estimate gradient and Fisher information matrix
        for (int i = 0; i < Base::polCount; ++i)
        {
            Base::diffObjFunc += Base::history_dlogsist[i] * (Base::history_J[i] - baseline);
        }
        Base::diffObjFunc /= Base::polCount;


        //--------- save value of distgrad
        Base::currentItStats->metaGradient = Base::diffObjFunc;
        //---------

        if (useDirection)
            Base::diffObjFunc = arma::normalise(Base::diffObjFunc);
        Base::diffObjFunc *= Base::step_length;

        //update meta distribution
        Base::dist.update(Base::diffObjFunc);


        //            std::cout << "diffObj: " << diffObjFunc[0].t();
        //            std::cout << "Parameters:\n" << std::endl;
        //            std::cout << dist.getParameters() << std::endl;

        b_num = 0.0;
        b_den = 0.0;
    }

private:
    double b_num, b_den;
    bool useDirection;
};

} //end namespace

#endif //PGPE_H_
