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

#ifndef REPS_H_
#define REPS_H_

#include "Agent.h"
#include "SampleManager.h"
#include "nonparametric/TabularPolicy.h"

#include <nlopt.hpp>

namespace ReLe
{

class TabularREPS: public Agent<FiniteAction, FiniteState>
{
public:
    TabularREPS();

    virtual void initEpisode(const FiniteState& state, FiniteAction& action);
    virtual void sampleAction(const FiniteState& state, FiniteAction& action);
    virtual void step(const Reward& reward, const FiniteState& nextState,
                      FiniteAction& action);
    virtual void endEpisode(const Reward& reward);
    virtual void endEpisode();

    virtual AgentOutputData* getAgentOutputData();
    virtual AgentOutputData* getAgentOutputDataEnd();

    virtual ~TabularREPS();

private:
    void updatePolicy();
    void updateSamples(size_t xn, double r);
    void resetSamples();

    double computeObjectiveFunction(const double* x, double* grad);

private:
    static double wrapper(unsigned int n, const double* x, double* grad,
                          void* o);

protected:
    virtual void init();

private:
    TabularPolicy policy;
    arma::vec thetaOpt;
    double etaOpt;

    int N;
    int currentIteration;
    double eps;

    //current an previous actions and states
    size_t x;
    unsigned int u;

    IdentityBasis phi;
    SampleManager<FiniteAction, FiniteState> s;

    nlopt::opt optimizator;

    //debug TODO togliere
    int iteration;

};

}

#endif /* REPS_H_ */
