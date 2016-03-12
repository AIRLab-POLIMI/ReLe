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
 * This class contains the settings of the Dam problem
 * and some functions to manage them.
 */
class DamSettings : public EnvironmentSettings
{
public:
    enum initType {RANDOM, RANDOM_DISCRETE};

    /*!
     * Constructor
     */
    DamSettings();

    /*!
     * Default settings initialization
     * \param settings the default settings
     */
    static void defaultSettings(DamSettings& settings);

    virtual ~DamSettings()
    {
    }

public:
    // Reservoir surface
    double S;
    // Water Demand
    double W_IRR;
    // Flooding threshold
    double H_FLO_U;
    double S_MIN_REL;
    double DAM_INFLOW_MEAN;
    double DAM_INFLOW_STD;
    double Q_MEF;
    double GAMMA_H2O;
    // Hydroelectric demand
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
 * This class implements the Dam problem environment.
 * The aim of this optimization problem is to decide
 * the amount of water to release in order to satisfy
 * conflicting objectives.
 *
 * References
 * ==========
 * [Castelletti, Pianosi, Restelli. A multiobjective reinforcement learning approach to water resources systems operation: Pareto frontier approximation in a single run. Water Resources Journal](http://onlinelibrary.wiley.com/doi/10.1002/wrcr.20295/abstract)
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
     * \param config settings of the environment
     */
    Dam(DamSettings& config);

    virtual ~Dam()
    {
        if (cleanConfig)
            delete config;
    }

    /*!
     * \see Environment::step
     */
    virtual void step(const DenseAction& action, DenseState& nextState,
                      Reward& reward) override;

    /*!
     * \see Environment::getInitialState
     */
    virtual void getInitialState(DenseState& state) override;

    /*!
     * Set the current state.
     * \param state current state
     */
    void setCurrentState(const DenseState& state);

    /*!
     * \see Environment::getSettings
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
