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

#include "policy_search/REPS/EpisodicREPS.h"

#include <iostream>

using namespace arma;
using namespace std;

namespace ReLe
{

EpisodicREPS::EpisodicREPS(ParametricNormal& dist,
                           ParametricPolicy<DenseAction, DenseState>& policy) :
    dist(dist), policy(policy)
{
    maxR = -std::numeric_limits<double>::infinity();
    etaOpt = 1;

    //default parameters
    eps = 0.5;
}

void EpisodicREPS::initEpisode(const DenseState& state, DenseAction& action)
{
    //Reset samples
    samples.clear();
    maxR = -std::numeric_limits<double>::infinity();

    theta = dist();
    policy.setParameters(theta);

    sampleAction(state, action);
}

void EpisodicREPS::sampleAction(const DenseState& state, DenseAction& action)
{
    vec& u = action;
    u = policy(state);
}

void EpisodicREPS::step(const Reward& reward, const DenseState& nextState,
                        DenseAction& action)
{
    updateSamples(reward[0]);
    theta = dist();
    policy.setParameters(theta);
    sampleAction(nextState, action);
}

void EpisodicREPS::endEpisode(const Reward& reward)
{
    double r = reward[0];

    updateSamples(r);
    updatePolicy();
    printStatistics();
}

void EpisodicREPS::endEpisode()
{
    updatePolicy();
    printStatistics();
}

EpisodicREPS::~EpisodicREPS()
{

}

void EpisodicREPS::updateSamples(double r)
{
    ParameterSample sample(theta, r);
    samples.push_back(sample);
}

void EpisodicREPS::updatePolicy()
{
    //optimize dual function
    std::vector<double> parameters(1, etaOpt);
    auto&& newParameters = optimizator.optimize(parameters);

    //update parameters
    etaOpt = newParameters.back();

    //Compute weights
    vec d(samples.size());
    for (unsigned int i = 0; i < samples.size(); i++)
    {
        double r = samples[i].r;
        d[i] = exp(r / etaOpt);
    }

    //Compute weights sums
    double dSum = sum(d);
    double d2Sum = sum(square(d));
    double Z = (dSum*dSum - d2Sum) / dSum;

    unsigned int thethaSize = policy.getParametersSize();

    //Compute mean
    vec mean(thethaSize, fill::zeros);
    for(unsigned int i = 0; i < d.size(); i++)
    {
        const vec& theta = samples[i].theta;
        mean += d[i]*theta;
    }

    mean /= dSum;

    //Compute covariance
    mat cov(thethaSize, thethaSize, fill::zeros);
    for(unsigned int i = 0; i < d.size(); i++)
    {
        const vec& theta = samples[i].theta;
        vec delta = theta - mean;
        cov += d[i]*delta*delta.t();
    }

    cov /= Z;

    //Update high level policy
    dist.setMeanAndCovariance(mean, cov);


}

double EpisodicREPS::computeObjectiveFunction(const double& eta, double& grad)
{
    double sum1 = 0;
    double sum2 = 0;

    double N = samples.size();

    for (auto& sample : samples)
    {
        double r = sample.r - maxR;
        sum1 += exp(r / eta);
        sum2 += exp(r / eta) * r;
    }

    sum1 /= N;
    sum2 /= N;

    grad = eps + log(sum1) - sum2 / (eta * sum1);
    return eta * eps + eta * log(sum1) + maxR;

}

double EpisodicREPS::wrapper(unsigned int n, const double* x, double* grad,
                             void* o)
{
    return reinterpret_cast<EpisodicREPS*>(o)->computeObjectiveFunction(*x,
            *grad);
}

void EpisodicREPS::init()
{
    //Init policy and parameters
    etaOpt = 1;

    //setup optimization algorithm
    optimizator = nlopt::opt(nlopt::algorithm::LD_MMA, 1);
    optimizator.set_min_objective(EpisodicREPS::wrapper, this);
    optimizator.set_xtol_rel(0.1);
    optimizator.set_ftol_rel(0.1);

    std::vector<double> lowerBounds(1, std::numeric_limits<double>::epsilon());
    optimizator.set_lower_bounds(lowerBounds);
}

void EpisodicREPS::printStatistics()
{
    cout << endl << endl << "### Episodic REPS ###";
    cout << endl << endl << "Using " << policy.getPolicyName() << " policy"
         << endl << endl;

    cout << "--- Parameters ---" << endl << endl;
    cout << "eps: " << eps << endl;

    cout << endl << endl << "--- Learning results ---" << endl << endl;
    cout << policy.printPolicy();
}

}

