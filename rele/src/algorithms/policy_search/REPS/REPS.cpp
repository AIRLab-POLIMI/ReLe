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

#include "policy_search/REPS/REPS.h"

using namespace arma;

namespace ReLe
{

TabularREPS::TabularREPS() :
			s(phi)
{
	x = 0;
	u = 0;
	etaOpt = 1;

	//default parameters
	N = 1;
	eps = 0.5;

	//sample iteration counter
	currentIteration = 0;
}

void TabularREPS::initEpisode(const FiniteState& state, FiniteAction& action)
{
	x = state.getStateN();
	u = policy(x);

	action.setActionN(u);

	resetSamples();
}

void TabularREPS::sampleAction(const FiniteState& state, FiniteAction& action)
{
	x = state.getStateN();
	u = policy(x);

	action.setActionN(u);
}

void TabularREPS::step(const Reward& reward, const FiniteState& nextState,
			FiniteAction& action)
{

	size_t xn = nextState.getStateN();
	unsigned int un = policy(xn);
	double r = reward[0];

	updateSamples(xn, r);

	if (currentIteration >= N)
	{
		currentIteration = 0;
		updatePolicy();
		resetSamples();
	}

	//update action and state
	x = xn;
	u = un;

	//set next action
	action.setActionN(u);
}

void TabularREPS::endEpisode(const Reward& reward)
{
	updatePolicy();
	printStatistics();
}

void TabularREPS::endEpisode()
{
	printStatistics();
}

TabularREPS::~TabularREPS()
{

}

void TabularREPS::updatePolicy()
{
	//optimize dual function
	std::vector<double> parameters(thetaOpt.begin(), thetaOpt.end());
	parameters.push_back(etaOpt);
	optimizator.optimize(parameters);

	//update parameters
	etaOpt = parameters.back();
	parameters.pop_back();
	thetaOpt = arma::vec(parameters);

	//compute new policy
	auto&& deltaOpt = s.getDelta(thetaOpt);
	for (size_t xi = 0; xi < task.finiteStateDim; xi++)
	{
		auto&& updater = policy.update(xi);
		while (updater.toFill())
		{
			unsigned int ui = updater.getCurrentState();
			updater << policy(xi, ui) * std::exp(deltaOpt(xi, ui) / etaOpt);
		}

		updater.normalize();
	}

}

void TabularREPS::updateSamples(size_t xn, double r)
{
	Sample<FiniteAction, FiniteState> sample(x, u, xn, r);
	s.addSample(sample);

	currentIteration++;
}

void TabularREPS::resetSamples()
{
	s.reset();
	currentIteration = 0;
}

double TabularREPS::computeObjectiveFunction(const double* x, double* grad)
{
	//Get state parameters
	const arma::vec theta(const_cast<double*>(x), this->thetaOpt.size(), false); //TODO check this
	double eta = x[this->thetaOpt.size()];

	auto&& delta = s.getDelta(theta);

	//compute needed sums
	double sum1 = 0;
	for (auto& sample : s)
	{
		sum1 += std::exp(eps + delta(sample.x, sample.u) / eta);
	}

	double sum2 = 0;
	for (auto& sample : s)
	{
		sum2 += std::exp(eps + delta(sample.x, sample.u) / eta)
					* delta(sample.x, sample.u) / std::pow(eta, 2);
	}

	arma::vec sum3(this->thetaOpt.size(), arma::fill::zeros);
	for (auto& sample : s)
	{
		sum3 += std::exp(eps + delta(sample.x, sample.u) / eta)
					* s.lambda(sample.x, sample.u);
	}

	//compute theta gradient
	arma::vec dTheta(grad, this->thetaOpt.size(), false);
	dTheta = eta * sum3 / sum1;

	//compute eta gradient
	double& dEta = grad[this->thetaOpt.size()];
	dEta = std::log(sum1) - sum2 / sum1;

	//compute dual function
	return eta * std::log(sum1 / N);
}

double TabularREPS::wrapper(unsigned int n, const double* x, double* grad,
			void* o)
{
	return reinterpret_cast<TabularREPS*>(o)->computeObjectiveFunction(x, grad);
}

void TabularREPS::init()
{
	//Init policy and parameters
	policy.init(task.finiteStateDim, task.finiteActionDim);
	thetaOpt = vec(task.finiteStateDim, fill::ones);
	etaOpt = 1;

	//setup basis function
	phi.setSize(task.finiteStateDim);

	//setup optimization algorithm
	optimizator = nlopt::opt(nlopt::algorithm::LD_LBFGS, thetaOpt.size() + 1);
	optimizator.set_min_objective(TabularREPS::wrapper, this);
	optimizator.set_xtol_rel(0.001);
}

void TabularREPS::printStatistics()
{
	cout << endl << endl << "### Tabular REPS ###";
	cout << endl << endl << "Using " << policy.getPolicyName() << " policy"
				<< endl << endl;

	cout << "--- Parameters ---" << endl << endl;
	cout << "N: " << N << endl;
	cout << "eps: " << eps << endl;

	cout << endl << endl << "--- Learning results ---" << endl << endl;
	cout << "- Policy" << endl;
	cout << policy.printPolicy();
}

}
