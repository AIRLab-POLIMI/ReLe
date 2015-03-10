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
#include <iomanip>

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
    arma::vec metaGradient;
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
        for (i = 0; i < stat.metaGradient.n_elem; ++i)
        {
            out << stat.metaGradient[i] << " ";
        }
        out << std::endl << ie << std::endl;
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
         unsigned int nbEpisodes, unsigned int nbPolicies,
         double step_length, bool baseline = true, int reward_obj = 0)
        : dist(dist), policy(policy),
          nbEpisodesToEvalPolicy(nbEpisodes), nbPoliciesToEvalMetap(nbPolicies),
          runCount(0), epiCount(0), polCount(0), df(1.0), step_length(step_length), useDirection(false),
          Jep (0.0), Jpol(0.0), rewardId(reward_obj),
          useBaseline(baseline)
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
        df  = 1.0;    //reset discount factor
        Jep = 0.0;    //reset J of current episode

        if (epiCount == 0)
        {
            //a new policy is considered
            Jpol = 0.0;

            //obtain new parameters
            arma::vec new_params = dist();
            //set to policy
            policy.setParameters(new_params);

            //create new policy individual
            PGPEPolicyIndividual polind(new_params, nbEpisodesToEvalPolicy);
            int dim = traces.size() - 1;
            traces[dim].individuals.push_back(polind);

            char f[555];
            sprintf(f, "PGPE_tracelog_r%d_p%d_e%d.txt", runCount, polCount, epiCount);
            //logger.fopen(f);
            //logger.print("1 1 1\n");
        }
        sampleAction(state, action);
        //logger.log(state);
        cAction = action;
    }

    void sampleAction(const StateC& state, ActionC& action)
    {
        typename action_type<ActionC>::type_ref u = action;
        u = policy(state);
    }


    template<class FiniteAction>
    void sampleAction(const StateC& state, FiniteAction& action)
    {
        unsigned int u = policy(state);
        action.setActionN(u);
    }

    void step(const Reward& reward, const StateC& nextState, ActionC& action)
    {
        //calculate current J value
        Jep += df * reward[rewardId];
        //update discount factor
        df *= Base::task.gamma;

        typename action_type<ActionC>::type_ref u = action;
        u = policy(nextState);

        //save action for logging operations
        cAction = action;
    }


    template<class FiniteAction>
    void step(const Reward& reward, const StateC& nextState, FiniteAction& action)
    {
        //calculate current J value
        Jep += df * reward[rewardId];
        //update discount factor
        df *= Base::task.gamma;

        unsigned int u = policy(nextState);
        action.setActionN(u);

        //save action for logging operations
        cAction = action;

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
        Jep = 0.0;

        //        std::cerr << "diffObjFunc: ";
        //        std::cerr << diffObjFunc[0].t();
        //        std::cout << "DLogDist(rho):";
        //        std::cerr << dlogdist.t();
        //        std::cout << "Jep:";
        //        std::cerr << Jep.t() << std::endl;

        //save actual policy performance
        int dim = traces.size() - 1;
        PGPEPolicyIndividual& polind = traces[dim].individuals[polCount];
        polind.Jvalues[epiCount] = Jep;

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

    inline void setNormalization(bool flag)
    {
        this->useDirection = flag;
    }

    inline bool isNormalized()
    {
        return this->useDirection;
    }

    void printStatistics(std::string filename)
    {
        std::ofstream out(filename, std::ios_base::out);
        out << std::setprecision(10);
        out << traces;
        out.close();
    }


protected:
    void init()
    {
        int dp = policy.getParametersSize();
        diffObjFunc = arma::vec(dp, arma::fill::zeros);
        history_dlogsist.assign(nbPoliciesToEvalMetap, diffObjFunc);
        history_J = arma::vec(nbPoliciesToEvalMetap, arma::fill::zeros);
    }

    void afterPolicyEstimate()
    {
        //average over episodes
        Jpol /= nbEpisodesToEvalPolicy;
        history_J[polCount] = Jpol;

        //compute gradient log distribution
        const arma::vec& theta = policy.getParameters();
        arma::vec dlogdist = dist.difflog(theta); //\nabla \log D(\theta|\rho)

        //compute baseline
        double norm2G2 = arma::norm(dlogdist,2);
        norm2G2 *= norm2G2;
        history_dlogsist[polCount] = dlogdist; //save gradients for late processing
        b_num += Jpol * norm2G2;
        b_den += norm2G2;


        //--------- save value of distgrad
        int dim = traces.size() - 1;
        PGPEPolicyIndividual& polind = traces[dim].individuals[polCount];
        polind.difflog = dlogdist;
        //---------

        ++polCount; //until now polCount policies have been analyzed
        epiCount = 0;
        Jpol = 0.0;
    }

    void afterMetaParamsEstimate()
    {

        //compute baseline
        double baseline = (b_den != 0 && useBaseline) ? b_num/b_den : 0.0;

        diffObjFunc.zeros();
        fisherMtx.zeros();
        //Estimate gradient and Fisher information matrix
        for (int i = 0; i < polCount; ++i)
        {
            diffObjFunc += history_dlogsist[i] * (history_J[i] - baseline);
        }
        diffObjFunc /= polCount;


        //--------- save value of distgrad
        int dim = traces.size() - 1;
        traces[dim].metaGradient = diffObjFunc;
        //---------

        if (useDirection)
            diffObjFunc = arma::normalise(diffObjFunc);
        diffObjFunc *= step_length;

        //update meta distribution
        dist.update(diffObjFunc);


        //            std::cout << "diffObj: " << diffObjFunc[0].t();
        //            std::cout << "Parameters:\n" << std::endl;
        //            std::cout << dist.getParameters() << std::endl;


        //reset counters and gradient
        polCount = 0;
        epiCount = 0;
        runCount++;

        b_num = 0.0;
        b_den = 0.0;

        // create statistic for first iteration
        PGPEIterationStats trace;
        trace.metaParams = dist.getParameters();
        traces.push_back(trace);
    }

private:
    DifferentiableDistribution& dist;
    ParametricPolicy<ActionC,StateC>& policy;
    unsigned int nbEpisodesToEvalPolicy, nbPoliciesToEvalMetap;
    unsigned int runCount, epiCount, polCount;
    double df, step_length;
    double Jep, Jpol;
    int rewardId;
    arma::vec diffObjFunc;
    double b_num, b_den;
    arma::mat fisherMtx;
    std::vector<arma::vec> history_dlogsist;
    arma::vec history_J;


    bool useDirection, useBaseline;
    PGPEStatistics traces;
    ActionC cAction;

};

} //end namespace

#endif //PGPE_H_
