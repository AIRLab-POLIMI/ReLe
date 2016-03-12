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

#ifndef SWINGPENDULUM_H_
#define SWINGPENDULUM_H_

#include "rele/core/DenseMDP.h"
#include "rele/utils/Range.h"

namespace ReLe
{

/*!
 * This class contains the settings of the pendulum problem
 * and some functions to manage them.
 */
class SwingUpSettings : public EnvironmentSettings
{
public:
    /*!
     * Constructor.
     */
    SwingUpSettings();

    /*!
     * Default settings initialization
     * \param settings the default settings
     */
    static void defaultSettings(SwingUpSettings& settings);

    virtual ~SwingUpSettings();

public:
    double stepTime;
    double mass, length, g, requiredUpTime, upRange;
    bool useOverRotated, random_start;

    Range actionRange;
    Range thetaRange;
    Range velocityRange;
    std::vector<double> actionList;

    virtual void WriteToStream(std::ostream& out) const;
    virtual void ReadFromStream(std::istream& in);
};

/*!
 * This class implements a task where a pendulum
 * has to be swinged controlling its rotation
 * and trying to reach the maximum height without
 * letting it fall.
 *
 * References
 * ==========
 * [Doya. Reinforcement Learning In Continuous Time and Space. Neural computation 12.1 (2000): 219-245.](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/Continuous.pdf)
 */
class DiscreteActionSwingUp: public DenseMDP
{
public:
    /*!
     * Constructor.
     */
    DiscreteActionSwingUp();

    /*!
     * Constructor.
     * \param config the initial settings
     */
    DiscreteActionSwingUp(SwingUpSettings& config);

    virtual ~DiscreteActionSwingUp()
    {
        if (cleanConfig)
            delete config;
    }

    /*!
     * \see Environment::step
     */
    void step(const FiniteAction& action, DenseState& nextState, Reward& reward) override;

    /*!
     * \see Environment::getInitialState
     */
    void getInitialState(DenseState& state) override;

    /*!
     * \see Environment::getSettings
     */
    inline const SwingUpSettings& getSettings() const
    {
        return *config;
    }

private:
    inline void adjustTheta(double& theta)
    {
        if (theta >= M_PI)
            theta -= 2.0 * M_PI;
        if (theta < -M_PI)
            theta += 2.0 * M_PI;
    }

private:
    double previousTheta, cumulatedRotation, overRotatedTime;
    bool overRotated;
    int upTime;
    //current state [theta, velocity]
    SwingUpSettings* config;
    bool cleanConfig;
};

}//end namespace

#endif // SWINGPENDULUM_H_
