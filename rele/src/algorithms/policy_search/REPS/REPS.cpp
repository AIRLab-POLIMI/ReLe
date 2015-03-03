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

#include <iostream>
#include <limits>

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
    auto&& newParameters = optimizator.optimize(parameters);

    cout << "----------------------------" << endl;

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

    cout << "$policy$" << endl;
    cout << policy(0,0) << ", " << policy(0,1) << endl;
    cout << policy(1,0) << ", " << policy(1,1) << endl;
    cout << policy(2,0) << ", " << policy(2,1) << endl;
    cout << policy(3,0) << ", " << policy(3,1) << endl;
    cout << policy(4,0) << ", " << policy(4,1) << endl;
    cout << "----------------------------" << endl;

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
    const vec theta(const_cast<double*>(x), this->thetaOpt.size(), false);
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
        sum2 += std::exp(eps + deltaxu / eta) * deltaxu
                / std::pow(eta, 2);
        sum3 += std::exp(eps + deltaxu / eta)
                * s.lambda(sample.x, sample.u);
    }

    cout << "sum1 " << sum1 << endl;
    cout << "sum2 " << sum2 << endl;
    cout << "sum3" << sum3.t();

    //compute theta gradient
    vec dTheta(grad, this->thetaOpt.size(), false);
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
    cout << "x = [" << x[0];
    for (int i = 1; i < n; i++)
    {
        cout << "," << x[i];
    }
    cout << "]" << endl;

    double value = reinterpret_cast<TabularREPS*>(o)->computeObjectiveFunction(
                       x, grad);

    cout << "grad = [" << grad[0];
    for (int i = 1; i < n; i++)
    {
        cout << "," << grad[i];
    }
    cout << "]" << endl;
    cout << "value " << value << endl;

    cout << "---" << endl;

    return value;
}

void TabularREPS::init()
{
    //Init policy and parameters
    policy.init(task.finiteStateDim, task.finiteActionDim);
    thetaOpt = vec(task.finiteStateDim, fill::zeros);
    etaOpt = 1;

    //setup basis function
    phi.setSize(task.finiteStateDim);

    //setup optimization algorithm
    optimizator = nlopt::opt(nlopt::algorithm::LD_LBFGS, thetaOpt.size() + 1);
    optimizator.set_min_objective(TabularREPS::wrapper, this);
    optimizator.set_xtol_rel(0.01);
    optimizator.set_ftol_rel(0.01);

    std::vector<double> lowerBounds(thetaOpt.size() +1, -std::numeric_limits<double>::infinity());
    lowerBounds.back() =  0.1;

    optimizator.set_lower_bounds(lowerBounds);
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
