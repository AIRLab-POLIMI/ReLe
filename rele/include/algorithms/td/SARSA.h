/*
 * SARSA.h
 *
 *  Created on: 12/gen/2015
 *      Author: dave
 */

#ifndef INCLUDE_ALGORITHMS_TD_SARSA_H_
#define INCLUDE_ALGORITHMS_TD_SARSA_H_

#include "Agent.h"
#include <armadillo>

namespace ReLe
{

class SARSA: public Agent<FiniteAction, FiniteState>
{
public:
	SARSA(std::size_t statesN, std::size_t actionN);
	virtual void initEpisode(const FiniteState& state, FiniteAction& action);
	virtual void sampleAction(const FiniteState& state, FiniteAction& action);
	virtual void step(const Reward& reward, const FiniteState& nextState,
				FiniteAction& action);
	virtual void endEpisode(const Reward& reward);
	virtual void endEpisode();

	void setAlpha(double alpha)
	{
		this->alpha = alpha;
	}

	void setEpsilon(double eps)
	{
		this->eps = eps;
	}

	virtual ~SARSA();

private:
	unsigned int policy(std::size_t x);
	void printStatistics();

private:
	//Action-value function
	arma::mat Q;

	//current an previous actions and states
	size_t x;
	unsigned int u;

	//algorithm parameters
	double alpha;
	double eps;

};

class SARSA_lambda: public Agent<FiniteAction, FiniteState>
{
public:
	SARSA_lambda(double lambda);
	virtual void initEpisode(const FiniteState& state, FiniteAction& action);
	virtual void sampleAction(const FiniteState& state, FiniteAction& action);
	virtual void step(const Reward& reward, const FiniteState& nextState,
				FiniteAction& action);
	virtual void endEpisode(const Reward& reward);
	virtual void endEpisode();

	virtual ~SARSA_lambda();

private:
	double lambda;
};

}

#endif /* INCLUDE_ALGORITHMS_TD_SARSA_H_ */
