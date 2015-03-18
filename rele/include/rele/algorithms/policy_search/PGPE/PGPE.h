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

#include "Agent.h"
#include "Distribution.h"
#include "Policy.h"
#include "Basics.h"
#include <cassert>
#include <iomanip>

#include "PGPEOutputData.h"

namespace ReLe
{


//Templates needed to handle different action types
template<class StateC>
void sampleActionWorker(const StateC& state, FiniteAction& action, ParametricPolicy<FiniteAction,StateC>& policy)
{
    unsigned int u = policy(state);
    action.setActionN(u);
}

template<class StateC, class ActionC>
void sampleActionWorker(const StateC& state, ActionC& action, ParametricPolicy<ActionC,StateC>& policy)
{
    typename action_type<ActionC>::type_ref u = action;
    u = policy(state);
}


//TODO: definire questo come PGPE
template<class ActionC, class StateC, class DistributionC, class AgentOutputC>
class BlackBoxAlgorithm: public Agent<ActionC, StateC>
{

    typedef Agent<ActionC, StateC> Base;
public:
    BlackBoxAlgorithm(DistributionC& dist, ParametricPolicy<ActionC, StateC>& policy,
                      unsigned int nbEpisodes, unsigned int nbPolicies,
                      double step_length, bool baseline = true, int reward_obj = 0)
        : dist(dist), policy(policy),
          nbEpisodesToEvalPolicy(nbEpisodes), nbPoliciesToEvalMetap(nbPolicies),
          runCount(0), epiCount(0), polCount(0), df(1.0), step_length(step_length), useDirection(false),
          Jep (0.0), Jpol(0.0), rewardId(reward_obj),
          useBaseline(baseline), output2LogReady(false)
    {
    }

    virtual ~BlackBoxAlgorithm()
    {}

    // Agent interface
public:
    virtual void initEpisode(const StateC& state, ActionC& action)
    {
        df  = 1.0;    //reset discount factor
        Jep = 0.0;    //reset J of current episode

        if (polCount == 0 && epiCount == 0)
        {
            currentItStats = new AgentOutputC(nbPoliciesToEvalMetap,
                                              policy.getParametersSize(),
                                              nbEpisodesToEvalPolicy);
            currentItStats->metaParams = dist.getParameters();
        }

        if (epiCount == 0)
        {
            //a new policy is considered
            Jpol = 0.0;

            //obtain new parameters
            arma::vec new_params = dist();
            //set to policy
            policy.setParameters(new_params);

            //--- create new policy individual
//            currentItStats->individuals.push_back(
//                PGPEPolicyIndividual(new_params, nbEpisodesToEvalPolicy));
            currentItStats->individuals[polCount].Pparams = new_params;
            //---
        }
        sampleAction(state, action);
    }

    virtual void initTestEpisode()
    {
        //obtain new parameters
        arma::vec new_params = dist();
        //set to policy
        policy.setParameters(new_params);
    }

    virtual void sampleAction(const StateC& state, ActionC& action)
    {
        sampleActionWorker(state, action, policy);
    }

    virtual void step(const Reward& reward, const StateC& nextState, ActionC& action)
    {
        //calculate current J value
        Jep += df * reward[rewardId];
        //update discount factor
        df *= Base::task.gamma;

        sampleAction(nextState, action);
    }

    virtual void endEpisode(const Reward& reward)
    {
        //add last contribute
        Jep += df * reward[rewardId];
        //perform remaining operation
        this->endEpisode();

    }

    virtual void endEpisode()
    {

        Jpol += Jep;

        //        std::cerr << "diffObjFunc: ";
        //        std::cerr << diffObjFunc[0].t();
        //        std::cout << "DLogDist(rho):";
        //        std::cerr << dlogdist.t();
        //        std::cout << "Jep:";
        //        std::cerr << Jep.t() << std::endl;

        //--- save actual policy performance
        currentItStats->individuals[polCount].Jvalues[epiCount] = Jep;
        //---


        //last episode is the number epiCount+1
        epiCount++;
        //check evaluation of actual policy
        if (epiCount == nbEpisodesToEvalPolicy)
        {
            afterPolicyEstimate();
        }


        if (polCount == nbPoliciesToEvalMetap)
        {
            //all policies have been evaluated
            //conclude gradient estimate and update the distribution
            afterMetaParamsEstimate();
        }
    }

    inline virtual void setNormalization(bool flag)
    {
        this->useDirection = flag;
    }

    inline virtual bool isNormalized()
    {
        return this->useDirection;
    }

    virtual AgentOutputData* getAgentOutputDataEnd()
    {
        if (output2LogReady)
        {
            //output is ready, activate flag
            output2LogReady = false;
            return currentItStats;
        }
        return nullptr;
    }


protected:
    virtual void init() = 0;
    virtual void afterPolicyEstimate() = 0;
    virtual void afterMetaParamsEstimate() = 0;

protected:
    DistributionC& dist;
    ParametricPolicy<ActionC,StateC>& policy;
    unsigned int nbEpisodesToEvalPolicy, nbPoliciesToEvalMetap;
    unsigned int runCount, epiCount, polCount;
    double df, step_length;
    double Jep, Jpol;
    int rewardId;
    arma::vec diffObjFunc;
    std::vector<arma::vec> history_dlogsist;
    arma::vec history_J;


    bool useDirection, useBaseline, output2LogReady;
    AgentOutputC* currentItStats;

};

template<class ActionC, class StateC>
class PGPE: public BlackBoxAlgorithm<ActionC, StateC, DifferentiableDistribution, PGPEIterationStats>
{
    typedef BlackBoxAlgorithm<ActionC, StateC, DifferentiableDistribution, PGPEIterationStats> Base;
public:
    PGPE(DifferentiableDistribution& dist, ParametricPolicy<ActionC, StateC>& policy,
         unsigned int nbEpisodes, unsigned int nbPolicies, double step_length,
         bool baseline = true, int reward_obj = 0)
        : BlackBoxAlgorithm<ActionC, StateC, DifferentiableDistribution, PGPEIterationStats>
        (dist, policy, nbEpisodes, nbPolicies, step_length, baseline, reward_obj),
        b_num(0.0), b_den(0.0)
    {
    }

    virtual ~PGPE() {}

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

        ++Base::polCount; //until now polCount policies have been analyzed
        Base::epiCount = 0;
        Base::Jpol = 0.0;
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

        if (Base::useDirection)
            Base::diffObjFunc = arma::normalise(Base::diffObjFunc);
        Base::diffObjFunc *= Base::step_length;

        //update meta distribution
        Base::dist.update(Base::diffObjFunc);


        //            std::cout << "diffObj: " << diffObjFunc[0].t();
        //            std::cout << "Parameters:\n" << std::endl;
        //            std::cout << dist.getParameters() << std::endl;


        //reset counters and gradient
        Base::polCount = 0;
        Base::epiCount = 0;
        Base::runCount++;

        b_num = 0.0;
        b_den = 0.0;

        Base::output2LogReady = true;
    }

private:
    double b_num, b_den;
};

} //end namespace

#endif //PGPE_H_
