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

#ifndef DAM_H_
#define DAM_H_

#include "rele/core/ContinuousMDP.h"

namespace ReLe
{

/*!
 * This class contains the settings of the DAM problem
 * and some functions to manage them.
 */
class DamSettings : public EnvironmentSettings
{
public:
    enum initType {RANDOM, RANDOM_DISCRETE};

    DamSettings();
    static void defaultSettings(DamSettings& settings);
    virtual ~DamSettings() {}

public:
    //! Reservoir surface
    double S;
    //! Water Demand
    double W_IRR;
    //! Flooding threshold
    double H_FLO_U;
    double S_MIN_REL;
    double DAM_INFLOW_MEAN;
    double DAM_INFLOW_STD;
    double Q_MEF;
    double GAMMA_H2O;
    //! Hydroelectric demand
    double W_HYD;
    double Q_FLO_D;
    double ETA;
    double G;

    bool penalize;

    virtual void WriteToStream(std::ostream& out) const;
    virtual void ReadFromStream(std::istream& in);

    initType initial_state_type;
};

/*!
 * This class implements the DAM problem environment.
 * The aim of this optimization problem is to decide
 * the amount of water to release in order to satisfy
 * conflicting objectives.
 * For further information see: (http://www.dhigroup.
 * com/upload/publications/mike11/Pedersen_RealTime.pdf)
 */
class Dam: public ContinuousMDP
{
public:
    /*!
     * Constructor.
     */
    Dam();
    /*!
     * Constructor.
     * \param settings of environment
     */
    Dam(DamSettings& config);

    virtual ~Dam()
    {
        if (cleanConfig)
            delete config;
    }

    /*!
     * Step function.
     * \param action to perform
     * \param state reached after the step
     * \param reward obtained with the step
     */
    virtual void step(const DenseAction& action, DenseState& nextState,
                      Reward& reward) override;
    /*!
     * Get the initial state.
     * \param initial state
     */
    virtual void getInitialState(DenseState& state) override;

    /*!
     * Set the current state.
     * \param current state
     */
    void setCurrentState(const DenseState& state);

    /*!
     * Getter.
     * Used to get environment setting
     * \return a reference to environment settings
     */
    inline const DamSettings& getSettings() const
    {
        return *config;
    }

private:
    DamSettings* config;
    bool cleanConfig;
    int nbSteps;
};

}

#endif /* DAM_H_ */
