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

//TODO: definire questo come PGPE
template<class ActionC, class StateC, class DistributionC>
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
          useBaseline(baseline)
    {
        // create statistic for first iteration
        PGPEIterationStats trace;
        trace.metaParams = dist.getParameters();
        traces.push_back(trace);
    }

    virtual ~BlackBoxAlgorithm()
    {}

    // Agent interface
public:
    virtual void initEpisode(const StateC& state, ActionC& action)
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
        }
        sampleAction(state, action);
    }

    virtual void sampleAction(const StateC& state, ActionC& action)
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

    virtual void step(const Reward& reward, const StateC& nextState, ActionC& action)
    {
        //calculate current J value
        Jep += df * reward[rewardId];
        //update discount factor
        df *= Base::task.gamma;

        typename action_type<ActionC>::type_ref u = action;
        u = policy(nextState);
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

    inline virtual void setNormalization(bool flag)
    {
        this->useDirection = flag;
    }

    inline virtual bool isNormalized()
    {
        return this->useDirection;
    }

    virtual void printStatistics(std::string filename)
    {
        std::ofstream out(filename, std::ios_base::out);
        out << std::setprecision(10);
        out << traces;
        out.close();
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


    bool useDirection, useBaseline;
    PGPEStatistics traces;


};

template<class ActionC, class StateC>
class PGPE: public BlackBoxAlgorithm<ActionC, StateC, DifferentiableDistribution>
{
    typedef BlackBoxAlgorithm<ActionC, StateC, DifferentiableDistribution> Base;
public:
    PGPE(DifferentiableDistribution& dist, ParametricPolicy<ActionC, StateC>& policy,
         unsigned int nbEpisodes, unsigned int nbPolicies, double step_length,
         bool baseline = true, int reward_obj = 0)
        : BlackBoxAlgorithm<ActionC, StateC, DifferentiableDistribution>(dist, policy, nbEpisodes, nbPolicies, step_length, baseline, reward_obj)
    {    }

protected:
    virtual void init()
    {
        int dp = Base::policy.getParametersSize();
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
        int dim = Base::traces.size() - 1;
        PGPEPolicyIndividual& polind = Base::traces[dim].individuals[Base::polCount];
        polind.difflog = dlogdist;
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
        int dim = Base::traces.size() - 1;
        Base::traces[dim].metaGradient = Base::diffObjFunc;
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

        // create statistic for first iteration
        PGPEIterationStats trace;
        trace.metaParams = Base::dist.getParameters();
        Base::traces.push_back(trace);
    }

private:
    double b_num, b_den;
};

} //end namespace

#endif //PGPE_H_
