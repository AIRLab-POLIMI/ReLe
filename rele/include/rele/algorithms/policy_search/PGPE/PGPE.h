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
#include <cassert>

namespace ReLe
{

template<class ActionC, class StateC>
class PGPE: public Agent<ActionC, StateC>
{

    typedef Agent<ActionC, StateC> Base;
public:
    PGPE(DifferentiableDistribution* dist, ParametricPolicy<DenseAction,DenseState>* policy,
         unsigned int nbEpisodes, unsigned int nbPolicies, double step_length)
        : dist(dist), policy(policy),
          nbEpisodesToEvalPolicy(nbEpisodes), nbPoliciesToEvalMetap(nbPolicies),
          epiCount(0), polCount(0), df(1.0), step_length(step_length)
    {}

    virtual ~PGPE()
    {}

    // Agent interface
public:
    void initEpisode(const StateC& state, ActionC& action)
    {
        df = 1.0;    //reset discount factor
        Jep.zeros(); //reset policy score
        if (epiCount == 0)
        {
            //obtain new parameters
            arma::vec new_params = (*dist)();
            //set to policy
            policy->setParameters(new_params);
        }
        sampleAction(state, action);
    }

    void sampleAction(const StateC& state, ActionC& action)
    {
        // TODO CONTROLLARE ASSEGNAMENTO
        arma::vec vect = (*policy)(state);
        for (int i = 0; i < vect.n_elem; ++i)
            action[i] = vect[i];
    }

    void step(const Reward& reward, const StateC& nextState, ActionC& action)
    {
        for (int i = 0; i < Jep.n_elem; ++i)
            Jep[i] += df * reward[i];
        df *= Base::task.gamma;
        // TODO CONTROLLARE ASSEGNAMENTO
        arma::vec vect = (*policy)(nextState);
        for (int i = 0; i < vect.n_elem; ++i)
            action[i] = vect[i];
    }

    virtual void endEpisode(const Reward& reward)
    {
        for (int i = 0; i < Jep.n_elem; ++i)
            Jep[i] += df * reward[i];
        this->endEpisode();
    }

    virtual void endEpisode()
    {

        const arma::vec& theta = policy->getParameters();
        arma::vec dlogdist = dist->difflog(theta); //\nabla \log D(\theta|\rho)
        for (int i = 0; i < Jep.n_elem; ++i)
            diffObjFunc[i] += dlogdist*Jep[i];


        //last episode is the number epiCount+1
        epiCount++;
        //check evaluation of actual policy
        if (epiCount == nbEpisodesToEvalPolicy)
        {
            ++polCount; //until now polCount policies have been analyzed
            epiCount = 0;
        }

        if (polCount == nbPoliciesToEvalMetap)
        {

            //update meta-distribution
            diffObjFunc[0] *= step_length/polCount;
            dist->Update(diffObjFunc[0]);


            //reset counters and gradient
            polCount = 0;
            epiCount = 0;
            for (int i = 0; i < Base::task.rewardDim; ++i)
                diffObjFunc[i].zeros();
        }
    }


protected:
    void init()
    {
        Jep = arma::vec(Base::task.rewardDim, arma::fill::zeros);
        for (int i = 0; i < Base::task.rewardDim; ++i)
            diffObjFunc.push_back(arma::vec(policy->getParametersSize(), arma::fill::zeros));

        assert(Base::task.rewardDim == 1);
    }

private:
    DifferentiableDistribution* dist;
    ParametricPolicy<ActionC,StateC>* policy;
    unsigned int nbEpisodesToEvalPolicy, nbPoliciesToEvalMetap;
    unsigned int epiCount, polCount;
    double df, step_length;
    arma::vec Jep;
    std::vector< arma::vec > diffObjFunc;

};

} //end namespace

#endif //PGPE_H_
