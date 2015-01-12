/*
 * SARSA.h
 *
 *  Created on: 12/gen/2015
 *      Author: dave
 */

#ifndef INCLUDE_ALGORITHMS_TD_SARSA_H_
#define INCLUDE_ALGORITHMS_TD_SARSA_H_

#include "Agent.h"

namespace ReLe
{

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
