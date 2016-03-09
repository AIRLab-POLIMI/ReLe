/*
 * rele,
 *
 *
 * Copyright (C) 2016 Davide Tateo
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

#ifndef INCLUDE_RELE_ALGORITHMS_STEP_RULES_LEARNINGRATE_H_
#define INCLUDE_RELE_ALGORITHMS_STEP_RULES_LEARNINGRATE_H_

#include "rele/core/Basics.h"

namespace ReLe
{

template<class ActionC, class StateC>
class LearningRate_
{
public:
	virtual double getLearningRate(StateC x, ActionC u) = 0;
	virtual void resetLearningRate(StateC x, ActionC u) = 0;

	virtual ~LearningRate_()
	{

	}
};

template<class ActionC, class StateC>
class ConstantLearningRate_ : public LearningRate_<ActionC, StateC>
{
public:
	ConstantLearningRate_(double alpha) : alpha(alpha)
	{

	}

	virtual double getLearningRate(StateC x, ActionC u) override
	{
		return alpha;
	}

	virtual void resetLearningRate(StateC x, ActionC u) override
	{

	}

	virtual ~ConstantLearningRate_()
	{

	}

private:
	double alpha;

};

typedef ConstantLearningRate_<FiniteAction, FiniteState> ConstantLearningRateStep;
typedef ConstantLearningRate_<FiniteAction, DenseState> ConstantLearningRateStepDense;

template<class ActionC, class StateC>
class DecayingLearningRate_ : public LearningRate_<ActionC, StateC>
{
public:
	DecayingLearningRate_(double initialAlpha, double omega = 1.0, double minAlpha = 0.0)
		: initialAlpha(initialAlpha), omega(omega), minAlpha(minAlpha)
	{
		t = 0;
	}

	virtual double getLearningRate(StateC x, ActionC u) override
	{
		t++;
		return initialAlpha/std::pow(static_cast<double>(t), omega);
	}

	virtual void resetLearningRate(StateC x, ActionC u) override
	{
		t = 0;
	}

	virtual ~DecayingLearningRate_()
	{

	}

private:
	double initialAlpha;
	double minAlpha;
	double omega;

	unsigned int t;

};

typedef DecayingLearningRate_<FiniteAction, FiniteState> DecayingLearningRateStep;
typedef DecayingLearningRate_<FiniteAction, DenseState> DecayingLearningRateStepDense;


}

#endif /* INCLUDE_RELE_ALGORITHMS_STEP_RULES_LEARNINGRATE_H_ */
