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

#ifndef INCLUDE_RELE_ENVIRONMENTS_ROCKY_H_
#define INCLUDE_RELE_ENVIRONMENTS_ROCKY_H_

#include "rele/core/ContinuousMDP.h"
#include "rele/utils/Range.h"

#include <vector>

namespace ReLe
{

/*!
 * This class implements an environment similar to the Pursuer problem.
 * In this task, Rocky has to reach the chicken moving in a
 * continuous MDP.
 */
class Rocky : public ContinuousMDP
{
public:
    /*!
     * Constructor.
     */
    Rocky();

    /*!
     * \see Environment::step
     */
    virtual void step(const DenseAction& action, DenseState& nextState,
                      Reward& reward) override;

    /*!
     * \see Environment::getInitialState
     */
    virtual void getInitialState(DenseState& state) override;

    enum StateComponents
    {
        // robot state
        x = 0,
        y,
        theta,
        // robot sensors
        energy,
        food,
        // rocky state
        xr,
        yr,
        thetar,
        // state size
        STATESIZE
    };

private:
    class Predictor
    {
    public:
        Predictor(double dt, Range limitX, Range limitY);
        void reset();
        void saveLastValues(double thetaM, double v);
        void predict(const DenseState& state, double& xhat, double& yhat, double& thetaDirhat);

    private:
        const double dt;

        // predictor state
        double thetaM;
        double v;

        // walls
        const Range limitX;
        const Range limitY;
    };

private:
    std::vector<arma::vec2> foodSpots;
    const double dt;
    const Range maxOmega;
    const Range maxV;
    const Range maxOmegar;
    const Range maxVr;
    const Range limitX;
    const Range limitY;
    const Range maxEnergy;

    // Predictor for rocky
    Predictor predictor;

    void computeReward(Reward& reward);
    void computeSensors(bool eat);
    void updateChickenPose(double v, double omega);
    void updateRockyPose(double vr, double omegar, double& xrabs,
                         double& yrabs);
    void computeRockyControl(double& vr, double& omegar);
};

}

#endif /* INCLUDE_RELE_ENVIRONMENTS_ROCKY_H_ */
