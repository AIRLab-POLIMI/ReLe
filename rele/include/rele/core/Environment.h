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

#ifndef environment_H_
#define environment_H_

#include <vector>

#include "rele/core/Basics.h"

namespace ReLe
{
/*!
 * The environment is the basic interface
 */
template<class ActionC, class StateC>
class Environment
{
    static_assert(std::is_base_of<Action, ActionC>::value, "Not valid Action class as template parameter");
    static_assert(std::is_base_of<State, StateC>::value, "Not a valid State class as template parameter");
public:

    /*!
     * Constructor
     */
    Environment()
        : settings(new EnvironmentSettings()), cleanSettings(true)
    {
    }

    /*!
     * Constructor
     * \param settings a pointer to the environment settings
     */
    Environment(EnvironmentSettings* settings)
        : settings(settings), cleanSettings(false)
    {
    }

    /*!
     * This function is called to execute an action on the environment. Must be implemented.
     * \param action the action to be executed at this time step
     * \param nextState the state reached after performing the action
     * \param reward the reward achieved by performing the last action on the environment
     */
    virtual void step(const ActionC& action, StateC& nextState,
                      Reward& reward) = 0;

    /*!
     * This function is called to get the initial environment state. Must be implemented.
     * \param state the initial state
     */
    virtual void getInitialState(StateC& state) = 0;

    /*!
     * Getter.
     * \return a const reference to the environment settings
     */
    inline const EnvironmentSettings& getSettings() const
    {
        return *settings;
    }

    /*!
     * Setter.
     * \param h the new environment horizon
     */
    void setHorizon(unsigned int h)
    {
        settings->horizon = h;
    }

    /*!
     * Destructor.
     */
    virtual ~Environment()
    {
        if (cleanSettings)
            delete settings;
    }

protected:
    /*!
     * Getter. To be used in derived classes.
     * \return a reference to the environment settings
     */
    inline EnvironmentSettings& getWritableSettings()
    {
        return *settings;
    }

protected:
    //! the environment settings
    EnvironmentSettings* settings;
private:
    bool cleanSettings;
};

}

#endif /* environment_H_ */
