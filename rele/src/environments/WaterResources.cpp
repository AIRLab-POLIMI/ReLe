/*
 * rele,
 *
 *
 * Copyright (C) 2017 Davide Tateo
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

#include "rele/environments/WaterResources.h"
#include "rele/utils/ArmadilloPDFs.h"
#include "rele/utils/Range.h"
#include "rele/utils/RandomGenerator.h"

using namespace std;

namespace ReLe
{

WaterResourcesSettings::WaterResourcesSettings()
    : mu(WaterResources::STATESIZE),
      Sigma(WaterResources::STATESIZE, WaterResources::STATESIZE),
      maxCapacity(WaterResources::STATESIZE),
      S(WaterResources::STATESIZE),
      h_flo(WaterResources::STATESIZE)
{
    WaterResourcesSettings::defaultSettings(*this);
}

void WaterResourcesSettings::defaultSettings(WaterResourcesSettings& settings)
{
    //Environment Parameters
    settings.gamma = 0.99;
    settings.stateDimensionality = WaterResources::STATESIZE;
    settings.actionDimensionality = WaterResources::STATESIZE;
    settings.rewardDimensionality = WaterResources::REWARDSIZE;
    settings.statesNumber = -1;
    settings.actionsNumber = -1;
    settings.isFiniteHorizon = false;
    settings.isAverageReward = false;
    settings.isEpisodic = false;
    settings.horizon = 100;

    //WaterResources Parameters
    //Noise parameters
    settings.mu[WaterResources::up] = 40;
    settings.mu[WaterResources::dn] = 20;
    settings.Sigma.eye();
    settings.Sigma(WaterResources::up, WaterResources::up) *= 16;
    settings.Sigma(WaterResources::dn, WaterResources::dn) *= 2;

    //Reservoirs parameters
    settings.maxCapacity[WaterResources::up] = 100;
    settings.maxCapacity[WaterResources::dn] = 100;
    settings.S[WaterResources::up] = 1;
    settings.S[WaterResources::dn] = 1;

    //Flooding related
    settings.h_flo[WaterResources::up] = 45;
    settings.h_flo[WaterResources::dn] = 50;

    //Irrigation parameters
    settings.w_irr = 60;

    //Power related
    settings.w_hyd = 4.36;
    settings.q_mef = 0.9;
    settings.eta = 1;
    settings.gamma_h20 = 1000;

    //Control delta
    settings.delta = 1.0;
}

WaterResourcesSettings::~WaterResourcesSettings()
{

}

void WaterResourcesSettings::WriteToStream(ostream& out) const
{
    //TODO [SERIALIZATION] implement
}

void WaterResourcesSettings::ReadFromStream(istream& in)
{
    //TODO [SERIALIZATION] implement
}



WaterResources::WaterResources()
    : ContinuousMDP(new WaterResourcesSettings()),
      cleanConfig(true),
      config(static_cast<WaterResourcesSettings*>(settings))
{
    currentState.set_size(this->getSettings().stateDimensionality);
}

WaterResources::WaterResources(WaterResourcesSettings& config)
    : ContinuousMDP(&config), cleanConfig(false), config(&config)
{
    currentState.set_size(this->getSettings().stateDimensionality);
}

void WaterResources::step(const DenseAction& action, DenseState& nextState,
                          Reward& reward)
{
    arma::vec eps = mvnrand(config->mu, config->Sigma);

    arma::vec& s = currentState;
    arma::vec V = (s + eps)/config->delta;
    arma::vec v(2);
    v[up] = std::max(s[up] + eps[up] - config->maxCapacity[up], 0.0)/config->delta;
    v[dn] = std::max(s[dn] + eps[dn] + V[up] - config->maxCapacity[dn], 0.0)/config->delta;

    Range limitUp(v[up], V[up]);
    Range limitDown(v[dn], V[dn]);

    arma::vec r(2);
    r[up] = limitUp.bound(action[up]);
    r[dn] = limitDown.bound(action[dn]);

    currentState[up] += (eps[up] - r[up])*config->delta;
    currentState[dn] += (eps[dn] + r[up] - r[dn])*config->delta;

    /*std::cout << "r: " << r.t() << std::endl;
    std::cout << "v: " << v.t() << std::endl;
    std::cout << "V: " << V.t() << std::endl;
    std::cout << "eps: " << eps.t() << std::endl;
    std::cout << "u: " << action.t() << std::endl;
    std::cout << "s: " << currentState.t() << std::endl;
    std::cout << "--------------------------------------" << std::endl;*/

    computeReward(reward, r);

    nextState = currentState;
}

void WaterResources::getInitialState(DenseState& state)
{
    currentState[up] = RandomGenerator::sampleUniform(0, config->maxCapacity[up]);
    currentState[dn] = RandomGenerator::sampleUniform(0, config->maxCapacity[dn]);

    state = currentState;
}

void WaterResources::computeReward(Reward& reward, const arma::vec& r)
{
    arma::vec& s = currentState;
    arma::vec h = s / config->S;
    double P = computePowerGeneration(h[up], r[up]);

    reward[flo_up] = -std::pow(std::max(h[up] - config->h_flo[up], 0.0), 2);
    reward[flo_dn] = -std::pow(std::max(h[dn] - config->h_flo[dn], 0.0), 2);
    reward[irr] = -std::pow(std::max(config->w_irr - r[dn], 0.0), 2);
    reward[hyd] = -std::max(config->w_hyd - P, 0.0);
}

double WaterResources::computePowerGeneration(double h, double r)
{
    const double g = 9.81;
    double q = std::max(r - config->q_mef, 0.0);

    return config->eta*g*config->gamma_h20*h * q / (3.6e6);
}

WaterResources::~WaterResources()
{
    if (cleanConfig)
        delete config;
}


}
