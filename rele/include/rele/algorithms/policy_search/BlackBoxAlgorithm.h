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

#ifndef BLACKBOXALGORITHM_H_
#define BLACKBOXALGORITHM_H_

#include "Agent.h"
#include "Distribution.h"
#include "Policy.h"
#include "Basics.h"
#include "policy_search/BlackBoxOutputData.h"
#include <cassert>
#include <iomanip>

namespace ReLe
{

template<class ActionC, class StateC, class DistributionC, class AgentOutputC>
class BlackBoxAlgorithm: public Agent<ActionC, StateC>
{

	typedef Agent<ActionC, StateC> Base;
public:
	BlackBoxAlgorithm(DistributionC& dist,
				ParametricPolicy<ActionC, StateC>& policy,
				unsigned int nbEpisodes, unsigned int nbPolicies,
				bool baseline = true, int reward_obj = 0) :
				dist(dist), policy(policy), nbEpisodesToEvalPolicy(nbEpisodes),
				nbPoliciesToEvalMetap(nbPolicies), runCount(0), epiCount(0),
				polCount(0), df(1.0), Jep(0.0), Jpol(0.0), rewardId(reward_obj),
				useBaseline(baseline), output2LogReady(false),
				currentItStats(nullptr)
	{
	}

	virtual ~BlackBoxAlgorithm()
	{
	}

	// Agent interface
public:
	virtual void initEpisode(const StateC& state, ActionC& action)
	{
		df = 1.0;    //reset discount factor
		Jep = 0.0;    //reset J of current episode

		if (polCount == 0 && epiCount == 0)
		{
			currentItStats = new AgentOutputC(nbPoliciesToEvalMetap,
						policy.getParametersSize(), nbEpisodesToEvalPolicy);
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

			//create new policy individual
			currentItStats->individuals[polCount].Pparams = new_params;

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

	virtual void step(const Reward& reward, const StateC& nextState,
				ActionC& action)
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
			epiCount = 0; //reset episode counter
			Jpol = 0.0; //reset policy value
			polCount++; //until now polCount policies have been analyzed
		}

		if (polCount == nbPoliciesToEvalMetap)
		{
			//all policies have been evaluated
			//conclude gradient estimate and update the distribution
			afterMetaParamsEstimate();

			//reset counters and gradient
			polCount = 0; //reset policy counter
			epiCount = 0; //reset episode counter
			runCount++; //update run counter
			output2LogReady = true; //output must be ready for log
		}
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
	ParametricPolicy<ActionC, StateC>& policy;
	unsigned int nbEpisodesToEvalPolicy, nbPoliciesToEvalMetap;
	unsigned int runCount, epiCount, polCount;
	double df;
	double Jep, Jpol;
	int rewardId;
	arma::vec history_J;

	bool useBaseline, output2LogReady;
	AgentOutputC* currentItStats;
};

template<class ActionC, class StateC, class DistributionC, class AgentOutputC>
class GradientBlackBoxAlgorithm: public BlackBoxAlgorithm<ActionC, StateC,
			DistributionC, AgentOutputC>
{
public:
	GradientBlackBoxAlgorithm(DistributionC& dist,
				ParametricPolicy<ActionC, StateC>& policy,
				unsigned int nbEpisodes, unsigned int nbPolicies,
				double step_length, bool baseline = true, int reward_obj = 0) :
				BlackBoxAlgorithm<ActionC, StateC, DistributionC, AgentOutputC>(
							dist, policy, nbEpisodes, nbPolicies, baseline,
							reward_obj),
				step_length(step_length)
	{
	}

	virtual ~GradientBlackBoxAlgorithm()
	{
	}

protected:

	double step_length;
	arma::vec diffObjFunc;
	std::vector<arma::vec> history_dlogsist;
};

#define USE_BBO_MEMBERS(AgentOutputClass)                                             \
	typedef BlackBoxAlgorithm<ActionC, StateC, DistributionC, AgentOutputClass> Base; \
    using Base::dist;                                                                 \
    using Base::policy;                                                               \
    using Base::nbEpisodesToEvalPolicy;                                               \
    using Base::nbPoliciesToEvalMetap;                                                \
    using Base::runCount;                                                             \
    using Base::epiCount;                                                             \
	using Base::polCount;                                                             \
    using Base::df;                                                                   \
    using Base::Jep;                                                                  \
    using Base::Jpol;                                                                 \
    using Base::rewardId;                                                             \
    using Base::history_J;                                                            \
    using Base::useBaseline;                                                          \
    using Base::output2LogReady;                                                      \
    using Base::currentItStats;

} //end namespace

#endif //BLACKBOXALGORITHM_H_
