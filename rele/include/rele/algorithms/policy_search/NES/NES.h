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

#include "policy_search/PGPE/PGPE.h"

namespace ReLe
{

template<class ActionC, class StateC>
class NES: public Agent<ActionC, StateC>
{

    typedef Agent<ActionC, StateC> Base;
public:
    NES(DifferentiableDistribution& dist, ParametricPolicy<ActionC, StateC>& policy,
        unsigned int nbEpisodes, unsigned int nbPolicies, double step_length,
        bool baseline = true, int reward_obj = 0)
        : dist(dist), policy(policy),
          nbEpisodesToEvalPolicy(nbEpisodes), nbPoliciesToEvalMetap(nbPolicies),
          runCount(0), epiCount(0), polCount(0), df(1.0), step_length(step_length),
          Jep (0.0), Jpol(0.0), rewardId(reward_obj),
          useBaseline(baseline),
          logger(false, "NES_tracelog_r0_p0_e0.txt")
    {
        // create statistic for first iteration
        PGPEIterationStats trace;
        trace.metaParams = dist.getParameters();
        traces.push_back(trace);
    }

    virtual ~NES()
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
            logger.fopen(f);
            logger.print("1 1 1\n");
        }
        sampleAction(state, action);
        logger.log(state);
        cAction = action;
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
        //save actual information
        logger.log(cAction,nextState,reward,0);


        //calculate current J value
        Jep += df * reward[rewardId];
        //update discount factor
        df *= Base::task.gamma;

        // TODO CONTROLLARE ASSEGNAMENTO PER RESTITUIRE L'AZIONE
        arma::vec vect = policy(nextState);
        for (int i = 0; i < vect.n_elem; ++i)
            action[i] = vect[i];

        //save action for logging operations
        cAction = action;
    }

    virtual void endEpisode(const Reward& reward)
    {
        //log last reward
        logger.log(reward);

        //add last contribute
        Jep += df * reward[rewardId];
        //perform remaining operation
        this->endEpisode();
    }

    virtual void endEpisode()
    {
        //logging operations
        logger.log();

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
        b_num = arma::vec(dp, arma::fill::zeros);
        b_den = arma::vec(dp, arma::fill::zeros);
        fisherMtx = arma::mat(dp,dp, arma::fill::zeros);
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
        history_dlogsist[polCount] = dlogdist; //save gradients for late processing
        arma::vec dlogdist2 = (dlogdist % dlogdist);
        b_num += Jpol * dlogdist2;
        b_den += dlogdist2;


        ++polCount; //until now polCount policies have been analyzed
        epiCount = 0;
        Jpol = 0.0;
    }

    void afterMetaParamsEstimate()
    {

        //compute baseline
        arma::vec baseline = b_num;
        if (useBaseline) {
            for (int i = 0, ie = baseline.n_elem; i < ie; ++i)
                if (b_den[i] != 0) {
                    baseline[i] /= b_den[i];
                } else {
                    baseline[i] = 0;
                }
        } else {
            baseline.zeros();
        }

        diffObjFunc.zeros();
        fisherMtx.zeros();
        //Estimate gradient and Fisher information matrix
        for (int i = 0; i < polCount; ++i)
        {
            diffObjFunc += (history_dlogsist[i] - baseline) * history_J[i];
            fisherMtx += history_dlogsist[i] * (history_dlogsist[i].t());
        }
        diffObjFunc /= polCount;
        fisherMtx /= polCount;

//        arma::mat tmp;
//        int rnk = arma::rank(fisherMtx);
//        if (rnk == fisherMtx.n_rows)
//        {
//            arma::mat H = arma::solve(fisherMtx, diffObjFunc);
//            tmp = diffObjFunc.t() * H;
//        } else {
//            std::cerr << "WARNING: Fisher Matrix is lower rank (rank = " << rnk << ")!!! Should be " << fisherMtx.n_rows << std::endl;
//            arma::mat H = arma::pinv(fisherMtx);
//            tmp = diffObjFunc.t() * (H * diffObjFunc);
//        }

//        double lambda = sqrt(tmp(0,0) / (4 * step_length));
//        lambda = std::max(lambda, 1e-8); // to avoid numerical problems
//        arma::vec nat_grad = arma::solve(fisherMtx, diffObjFunc) / (2 * lambda);

        arma::mat tmp;
        int rnk = arma::rank(fisherMtx);
        if (rnk == fisherMtx.n_rows)
        {
            arma::mat H = arma::solve(fisherMtx, diffObjFunc);
            tmp = H;
        } else {
            std::cerr << "WARNING: Fisher Matrix is lower rank (rank = " << rnk << ")!!! Should be " << fisherMtx.n_rows << std::endl;
            arma::mat H = arma::pinv(fisherMtx);
            tmp = H * diffObjFunc;
        }

        arma::vec nat_grad = tmp*step_length;


        //--------- save value of distgrad
        int dim = traces.size() - 1;
        traces[dim].metaGradient = nat_grad;
        //---------

        //update meta distribution
        dist.update(nat_grad);


        std::cout << "nat_grad: " << nat_grad.t();
        std::cout << "Parameters:\n" << std::endl;
        std::cout << dist.getParameters() << std::endl;


        //reset counters and gradient
        polCount = 0;
        epiCount = 0;
        runCount++;
        for (int i = 0, ie = diffObjFunc.n_elem; i < ie; ++i)
        {
            //diffObjFunc[i] = 0.0;
            b_num[i] = 0.0;
            b_den[i] = 0.0;
        }

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
    double df, Jep, Jpol, step_length;
    int rewardId;
    bool useBaseline;
    arma::vec diffObjFunc, b_num, b_den;
    arma::mat fisherMtx;
    std::vector<arma::vec> history_dlogsist;
    arma::vec history_J;



    PGPEStatistics traces;
    Logger<ActionC,StateC> logger;
    ActionC cAction;


};

} //end namespace

#endif //NES_H_
