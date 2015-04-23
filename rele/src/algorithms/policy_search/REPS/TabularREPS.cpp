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

#include <iostream>
#include <limits>
#include "policy_search/REPS/TabularREPS.h"
#include "policy_search/REPS/REPSOutputData.h"

using namespace arma;

//#define DEBUG_REPS

namespace ReLe
{

TabularREPS::TabularREPS(DenseFeatures_<size_t>& phi) :
    phi(phi), s(phi)
{
    x = 0;
    u = 0;
    etaOpt = 1;

    //default parameters
    N = 100;
    eps = 0.5;

    //sample iteration counter
    currentIteration = 0;

    //TODO levami!
    iteration = 0;
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
    updateSamples(x, reward[0]); //TODO check this update
    updatePolicy();
}

void TabularREPS::endEpisode()
{
    updatePolicy();
}

AgentOutputData* TabularREPS::getAgentOutputData()
{
    if(currentIteration == 0)
    {
        return new TabularREPSOutputData(N, eps, policy.printPolicy(), false);
    }
    else
    {
        return nullptr;
    }
}

AgentOutputData* TabularREPS::getAgentOutputDataEnd()
{
    return new TabularREPSOutputData(N, eps, policy.printPolicy(), true);
}

TabularREPS::~TabularREPS()
{

}

void TabularREPS::updatePolicy()
{

    //TODO levami!
    iteration = 0;

    //optimize dual function
    std::vector<double> parameters(thetaOpt.begin(), thetaOpt.end());
    parameters.push_back(etaOpt);
    auto&& newParameters = optimizator.optimize(parameters);

#ifdef DEBUG_REPS
    cout << "----------------------------" << endl;
#endif

    //update parameters
    etaOpt = newParameters.back();
    newParameters.pop_back();
    thetaOpt = vec(newParameters);

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
    const vec theta(const_cast<double*>(x), this->thetaOpt.size(), true);
    double eta = x[this->thetaOpt.size()];

    auto&& delta = s.getDelta(theta);

    //compute needed sums
    double sum1 = 0;
    double sum2 = 0;
    vec sum3(this->thetaOpt.size(), fill::zeros);

    for (auto& sample : s)
    {
        double deltaxu = delta(sample.x, sample.u);
        sum1 += std::exp(eps + deltaxu / eta);
        sum2 += std::exp(eps + deltaxu / eta) * deltaxu / std::pow(eta, 2);
        sum3 += std::exp(eps + deltaxu / eta) * s.lambda(sample.x, sample.u);
    }

#ifdef DEBUG_REPS
    cout << "sum1 " << sum1 << endl;
    cout << "sum2 " << sum2 << endl;
    cout << "sum3" << sum3.t();
#endif

    //Avoid NaN
    if(sum1 < std::numeric_limits<double>::epsilon())
    {
        for(int i = 0; i < this->thetaOpt.size() + 1; i++)
        {
            grad[i] = 0;
        }

        return -std::numeric_limits<double>::infinity();
    }

    //compute theta gradient
    vec dTheta(grad, this->thetaOpt.size(), false);
    dTheta = eta * sum3 / sum1;

    //compute eta gradient
    double& dEta = grad[this->thetaOpt.size()];
    dEta = std::log(sum1) - sum2 / sum1;

#ifdef DEBUG_REPS
    //TODO levami
    iteration++;
    cout << "iteration: " << iteration << endl;
#endif

    //compute dual function
    return eta * std::log(sum1 / N);
}

double TabularREPS::wrapper(unsigned int n, const double* x, double* grad,
                            void* o)
{
#ifdef DEBUG_REPS
    cout << "x = [" << x[0];
    for (int i = 1; i < n; i++)
    {
        cout << "," << x[i];
    }
    cout << "]" << endl;
#endif

    double value = reinterpret_cast<TabularREPS*>(o)->computeObjectiveFunction(
                       x, grad);

#ifdef DEBUG_REPS
    cout << "grad = [" << grad[0];
    for (int i = 1; i < n; i++)
    {
        cout << "," << grad[i];
    }
    cout << "]" << endl;
    cout << "value " << value << endl;

    cout << "---" << endl;
#endif

    return value;
}

void TabularREPS::init()
{
    //Init policy and parameters
    policy.init(task.finiteStateDim, task.finiteActionDim);
    thetaOpt = vec(task.finiteStateDim, fill::zeros);
    etaOpt = 1;

    //setup optimization algorithm
    optimizator = nlopt::opt(nlopt::algorithm::LD_MMA, thetaOpt.size() + 1);
    optimizator.set_min_objective(TabularREPS::wrapper, this);
    optimizator.set_xtol_rel(1e-8);
    optimizator.set_ftol_rel(1e-12);
    optimizator.set_maxeval(300 * 10);

    std::vector<double> lowerBounds(thetaOpt.size() + 1,
                                    -std::numeric_limits<double>::infinity());
    lowerBounds.back() = std::numeric_limits<double>::epsilon();

    optimizator.set_lower_bounds(lowerBounds);
}

}
