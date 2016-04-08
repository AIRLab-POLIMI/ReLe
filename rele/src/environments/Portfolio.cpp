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

#include "rele/environments/Portfolio.h"

#include "rele/utils/RandomGenerator.h"
#include <cassert>

using namespace std;

namespace ReLe
{

PortfolioSettings::PortfolioSettings()
{
    PortfolioSettings::defaultSettings(*this);
}

void PortfolioSettings::defaultSettings(PortfolioSettings& settings)
{
    //Environment Parameters
    settings.gamma = 1.0;
    settings.stateDimensionality = N_STEPS + 2;
    settings.actionDimensionality = -1;
    settings.rewardDimensionality = 1;
    settings.statesNumber = -1;
    settings.actionsNumber = 2;
    settings.isFiniteHorizon = false;
    settings.isAverageReward = false;
    settings.isEpisodic = false;
    settings.horizon = T_STEPS;

    //Portfolio Parameters
    // time dependent variables
    settings.t = 0;
    settings.rNL = RNL_HIGH;
    settings.T_rNL = 0;
    settings.retL = 0;
    settings.retNL = 0;
    settings.T_Ret_Inv = 0;

    // time independent variables
    settings.T = T_STEPS;
    settings.n = N_STEPS;
    settings.alpha = ALPHA;
    settings.P_Risk = P_RISK;
    settings.P_Switch = P_SWITCH;
    settings.rL = RL;
    settings.rNL_High = RNL_HIGH;
    settings.rNL_Low = RNL_LOW;

    // initial state
    settings.initial_state = 1.0;
}

void PortfolioSettings::WriteToStream(ostream &out) const
{
    EnvironmentSettings::writeToStream(out);
    out << std::endl;
}

void PortfolioSettings::ReadFromStream(istream &in)
{
    EnvironmentSettings::readFromStream(in);
}


///////////////////////////////////////////////////////////////////////////////////////
/// PORTFOLIO ENVIRONMENTS
///////////////////////////////////////////////////////////////////////////////////////

Portfolio::Portfolio() :
    DenseMDP(new PortfolioSettings()), cleanConfig(true), config(static_cast<PortfolioSettings*>(settings))
{
    currentState.set_size(this->getSettings().stateDimensionality);
}

Portfolio::Portfolio(PortfolioSettings& config):
    DenseMDP(&config), cleanConfig(false), config(&config)
{
    currentState.set_size(this->getSettings().stateDimensionality);
}

void Portfolio::step(const FiniteAction& action, DenseState& nextState, Reward& reward)
{
    PortfolioSettings& nlsconfig = *config;

    // time dependent variables
    unsigned int t = nlsconfig.t;
    double rNL = nlsconfig.rNL;
    double T_rNL = nlsconfig.T_rNL;
    double retL = 0;
    double retNL = 0;
    double T_Ret_Inv = nlsconfig.T_Ret_Inv;

    // time independent variables
    unsigned int T = nlsconfig.T;
    unsigned int n = nlsconfig.n;
    double alpha = nlsconfig.alpha;
    double P_Risk = nlsconfig.P_Risk;
    double P_Switch = nlsconfig.P_Switch;
    double rL = nlsconfig.rL;
    double rNL_High = nlsconfig.rNL_High;
    double rNL_Low = nlsconfig.rNL_Low;


    // next instant
    t++;

    // NL interest rate can change at each step with Pr(PSWITCH)
    double random = RandomGenerator::sampleUniform(0,1);
    if (random <= P_Switch)
        rNL = (rNL == rNL_High) ? rNL_Low : rNL_High;

    // estimate of E[rNL]
    T_rNL += rNL;
    nextState[n + 1] = rNL - T_rNL / t;

    // L capital di preserved
    nextState[0] = currentState[0];

    // if a NL investment is mature then ALPHA returns available in L
    if (currentState[1] > 0)
    {
        nextState[0] += alpha;

        // the NL investment can be earned with interest or lost with Pr(PRISK)
        double random = RandomGenerator::sampleUniform(0,1);
        if (random > P_Risk)
            retNL = rNL * n * currentState[1];
    }

    // all the others NL investments mature
    for (unsigned int i = 1; i < n; ++i)
        nextState[i] = currentState[i + 1];

    unsigned int u = action.getActionN();
    assert(u == 0 || u == 1);


    // if the action is "invest" and the L capital is enough then ALPHA can be invested in NL
    if (u == 1 && nextState[0] >= alpha)
    {
        nextState[0] -= alpha;
        nextState[n] = alpha;
    }
    else
        nextState[n] = 0;// investment not made (action "not invest" or lack of L capital)

    // L interest is earned
    retL = rL * nextState[0];

    // reward is the logarithm of the total return from the current step
    reward[0] = retL + retNL;

    // update the total return from the investment of the current episode
    T_Ret_Inv += retL + retNL;

    // after T steps the process ends
    bool absorbing = (t >= T) ? true : false;
    nextState.setAbsorbing(absorbing);

    // std::cout << "t" << "\t" << "rNL" << "\t" << "T_rNL" << " \t" << "retL" << "\t" << "retNL" << "\t" << "T_Ret_Inv" << std::endl;
    // std::cout << t << "\t" << rNL << "\t" << T_rNL << "\t" << retL << "\t" << retNL << "\t" << T_Ret_Inv << std::endl;

    // time dependent variables are saved in config or restored to their default values
    if (!absorbing)
    {
        nlsconfig.t = t;
        nlsconfig.rNL = rNL;
        nlsconfig.T_rNL = T_rNL;
        nlsconfig.retL = retL;
        nlsconfig.retNL = retNL;
        nlsconfig.T_Ret_Inv = T_Ret_Inv;
    }
    else
        defaultValues();
    currentState = nextState;
}

void Portfolio::getInitialState(DenseState& state)
{
    const PortfolioSettings& nlsconfig = *config;
    currentState.zeros();
    currentState[0] = nlsconfig.initial_state;
    currentState.setAbsorbing(false);
    state = currentState;
}

void Portfolio::defaultValues()
{
    PortfolioSettings& nlsconfig = *config;
    // time dependent variables
    nlsconfig.t = 0;
    nlsconfig.rNL = RNL_HIGH;
    nlsconfig.T_rNL = 0;
    nlsconfig.retL = 0;
    nlsconfig.retNL = 0;
    nlsconfig.T_Ret_Inv = 0;
}


}  //end namespace
