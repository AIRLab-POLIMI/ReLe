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

class PGPEPolicyIndividual
{
public:
    arma::vec Pparams;  //policy parameters
    arma::vec Jvalues;  //policy evaluation (n evaluations for each policy)
    arma::mat difflog;

public:
    PGPEPolicyIndividual(arma::vec& polp, int nbEval)
        :Pparams(polp), Jvalues(nbEval), difflog(polp.n_elem, nbEval)
    {}

    friend std::ostream& operator<<(std::ostream& out, PGPEPolicyIndividual& stat)
    {
        int nparams = stat.Pparams.n_elem;
        int nepisodes = stat.Jvalues.n_elem;
        out << nparams;
        for (int i = 0; i < nparams; ++i)
            out << " " << stat.Pparams[i];
        out << std::endl;
        out << nepisodes;
        for (int i = 0; i < nepisodes; ++i)
            out << " " << stat.Jvalues[i];
        out << std::endl;
        for (int i = 0; i < nepisodes; ++i)
        {
            for (int j = 0; j < nparams; ++j)
            {
                out << stat.difflog(j,i) << " ";
            }
            out << std::endl;
        }
        return out;
    }

    friend std::istream& operator>>(std::istream& in, PGPEPolicyIndividual& stat)
    {
        int i, nbPolPar, nbEval;
        in >> nbPolPar;
        stat.Pparams = arma::vec(nbPolPar);
        for (i = 0; i < nbPolPar; ++i)
            in >> stat.Pparams[i];
        in >> nbEval;
        stat.Jvalues = arma::vec(nbEval);
        for (i = 0; i < nbEval; ++i)
            in >> stat.Jvalues[i];
        stat.difflog = arma::mat(nbPolPar, nbEval);
        for (int i = 0; i < nbPolPar; ++i)
        {
            for (int j = 0; j < nbEval; ++j)
            {
                in >> stat.difflog(i,j);
            }
        }
        return in;
    }
};

struct PGPEIterationStats
{
    arma::vec metaParams;
    std::vector<PGPEPolicyIndividual> individuals;

public:
    friend std::ostream& operator<<(std::ostream& out, PGPEIterationStats& stat)
    {
        int i, ie = stat.individuals.size();
        out << stat.metaParams.n_elem << " ";
        for (i = 0; i < stat.metaParams.n_elem; ++i)
        {
            out << stat.metaParams[i] << " ";
        }
        out << std::endl;
        out << ie << std::endl;
        for (i = 0; i < ie; ++i)
        {
            out << stat.individuals[i];
        }
        return out;
    }

    void addIndividual(PGPEPolicyIndividual& individual)
    {
        individuals.push_back(individual);
    }

};

class PGPEStatistics : public std::vector<PGPEIterationStats>
{
public:
    friend std::ostream& operator<<(std::ostream& out, PGPEStatistics& stat)
    {
        int i, ie = stat.size();
        out << ie << std::endl;
        for (i = 0; i < ie; ++i)
        {
            out << stat[i];
        }
        return out;
    }
};

template<class ActionC, class StateC>
class PGPE: public Agent<ActionC, StateC>
{

    typedef Agent<ActionC, StateC> Base;
public:
    PGPE(DifferentiableDistribution& dist, ParametricPolicy<ActionC, StateC>& policy,
         unsigned int nbEpisodes, unsigned int nbPolicies, double step_length)
        : dist(dist), policy(policy),
          nbEpisodesToEvalPolicy(nbEpisodes), nbPoliciesToEvalMetap(nbPolicies),
          epiCount(0), polCount(0), df(1.0), step_length(step_length), useDirection(false)
    {
        // create statistic for first iteration
        PGPEIterationStats trace;
        trace.metaParams = dist.getParameters();
        traces.push_back(trace);
    }

    virtual ~PGPE()
    {}

    // Agent interface
public:
    void initEpisode(const StateC& state, ActionC& action)
    {
        std::cout << "##InitEpisode" << std::endl;
        df = 1.0;    //reset discount factor
        if (epiCount == 0)
        {
            //obtain new parameters
            arma::vec new_params = dist();
            //set to policy
            policy.setParameters(new_params);

            //create new policy individual
            PGPEPolicyIndividual polind(new_params, nbEpisodesToEvalPolicy);
            int dim = traces.size() - 1;
            traces[dim].individuals.push_back(polind);
        }
        sampleAction(state, action);
    }

    void sampleAction(const StateC& state, ActionC& action)
    {
        // TODO CONTROLLARE ASSEGNAMENTO
        arma::vec vect = policy(state);
        action.copy_vec(vect);
//        for (int i = 0; i < vect.n_elem; ++i)
//            action[i] = vect[i];
    }

    void step(const Reward& reward, const StateC& nextState, ActionC& action)
    {
        for (int i = 0; i < Jep.n_elem; ++i)
            Jep[i] += df * reward[i];
        df *= Base::task.gamma;
        // TODO CONTROLLARE ASSEGNAMENTO
        arma::vec vect = policy(nextState);
        for (int i = 0; i < vect.n_elem; ++i)
            action[i] = vect[i];
    }

    virtual void endEpisode(const Reward& reward)
    {
        std::cout << "###End Episode (reward " << reward[0] << ")" << std::endl;
        for (int i = 0; i < Jep.n_elem; ++i)
            Jep[i] += df * reward[i];
        this->endEpisode();
    }

    virtual void endEpisode()
    {


        std::cout << "###End Episode (no reward)" << std::endl;

        const arma::vec& theta = policy.getParameters();
        arma::vec dlogdist = dist.difflog(theta); //\nabla \log D(\theta|\rho)
        std::cerr << "diffObjFunc: ";
        std::cerr << diffObjFunc[0].t();
        std::cout << "DLogDist(rho):";
        std::cerr << dlogdist.t() << std::endl;
        std::cout << "Jep:";
        std::cerr << Jep.t() << std::endl;

        //save actual policy performance
        int dim = traces.size() - 1;
        PGPEPolicyIndividual& polind = traces[dim].individuals[polCount];
        polind.Jvalues[epiCount] = Jep[0];


        for (int i = 0; i < Jep.n_elem; ++i)
            diffObjFunc[i] += dlogdist*Jep[i];

        std::cerr << "................\n";
        std::cerr << traces;
        std::cerr << "................\n";


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
            //HERE a new run starts


            //update meta-distribution
            diffObjFunc[0] *= step_length/polCount;
            if (useDirection)
                diffObjFunc[0] = arma::normalise(diffObjFunc[0]);
            dist.update(diffObjFunc[0]);

            std::cout << "diffObj: " << diffObjFunc[0].t();
            std::cout << "Parameters:\n" << std::endl;
            std::cout << dist.getParameters() << std::endl;


            //reset counters and gradient
            polCount = 0;
            epiCount = 0;
            for (int i = 0; i < Base::task.rewardDim; ++i)
                diffObjFunc[i].zeros();

            // create statistic for first iteration
            PGPEIterationStats trace;
            trace.metaParams = dist.getParameters();
            traces.push_back(trace);
        }
    }

    inline void setNormalization(bool flag)
    {
        this->useDirection = flag;
    }

    inline bool isNormalized()
    {
        return this->useDirection;
    }


protected:
    void init()
    {
        Jep = arma::vec(Base::task.rewardDim, arma::fill::zeros);
        std::cerr << policy.getParametersSize() << std::endl;
        for (int i = 0; i < Base::task.rewardDim; ++i)
            diffObjFunc.push_back(arma::vec(policy.getParametersSize(), arma::fill::zeros));

        assert(Base::task.rewardDim == 1);
    }

private:
    DifferentiableDistribution& dist;
    ParametricPolicy<ActionC,StateC>& policy;
    unsigned int nbEpisodesToEvalPolicy, nbPoliciesToEvalMetap;
    unsigned int epiCount, polCount;
    double df, step_length;
    arma::vec Jep;
    std::vector<arma::vec> diffObjFunc;
    bool useDirection;
    PGPEStatistics traces;

};

} //end namespace

#endif //PGPE_H_
