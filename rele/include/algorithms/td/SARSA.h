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

class SARSA : public Agent<FiniteAction, FiniteState>
{
public:
	SARSA(std::size_t statesN, std::size_t actionN);
	virtual void initEpisode();
	virtual void sampleAction(const FiniteState& state, FiniteAction& action);
	virtual void step(const Reward& reward, const FiniteState& nextState);
	virtual void endEpisode(const Reward& reward);

	void setAlpha(double alpha)
	{
		this->alpha = alpha;
	}

	void setGamma(double gamma)
	{
		this->gamma = gamma;
	}

	virtual ~SARSA();

private:
	unsigned int policy(std::size_t x);

private:
	//Action-value function
	arma::mat Q;

	//current an previous actions and states
	size_t x;
	unsigned int u;

	//algorithm parameters
	int t;
	double alpha;
	double gamma;

};

class SARSA_lambda : public Agent<FiniteAction, FiniteState>
{
public:
	SARSA_lambda(double lambda);
	virtual void initEpisode();
	virtual void sampleAction(const FiniteState& state, FiniteAction& action);
	virtual void step(const Reward& reward, const FiniteState& nextState);
	virtual void endEpisode(const Reward& reward);

	virtual ~SARSA_lambda();

private:
	double lambda;
};

}

#endif /* INCLUDE_ALGORITHMS_TD_SARSA_H_ */
