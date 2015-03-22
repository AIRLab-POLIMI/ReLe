/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo
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

#ifndef EPISODICREPS_H_
#define EPISODICREPS_H_

#include "Agent.h"
#include "DifferentiableNormals.h"
#include "nonparametric/TabularPolicy.h"

#include <nlopt.hpp>

/*namespace ReLe
{
class EpisodicREPS: public Agent<DenseAction, DenseState>
{

public:
    EpisodicREPS(ParametricNormal& dist, ParametricPolicy<DenseAction, DenseState>& policy);

    virtual void initEpisode(const DenseState& state, DenseAction& action);
    virtual void sampleAction(const DenseState& state, DenseAction& action);
    virtual void step(const Reward& reward, const DenseState& nextState,
                      DenseAction& action);
    virtual void endEpisode(const Reward& reward);
    virtual void endEpisode();

    virtual void initTestEpisode();

    virtual AgentOutputData* getAgentOutputDataEnd();

    inline void setEps(double eps)
    {
        this->eps = eps;
    }

    virtual ~EpisodicREPS();

private:
    void updatePolicy();

    double computeObjectiveFunction(const double& x, double& grad);

private:
    static double wrapper(unsigned int n, const double* x, double* grad,
                          void* o);
    void updateSamples(double r);

protected:
    virtual void init();

private:
    ParametricNormal& dist;
    ParametricPolicy<DenseAction, DenseState>& policy;
    double etaOpt;

    double eps;

    //Last parameters
    arma::vec theta;

    std::vector<ParameterSample> samples;
    double maxR;

    nlopt::opt optimizator;

};

}*/

#endif /* EPISODICREPS_H_ */
