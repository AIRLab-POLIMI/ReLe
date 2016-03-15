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

#ifndef NLS_H_
#define NLS_H_

#include "rele/core/ContinuousMDP.h"

namespace ReLe
{

/*!
 * This class contains the settings of the NLS problem
 * and some functions to manage them.
 */
class NLSSettings : public EnvironmentSettings
{
public:
    /*!
     * Constructor.
     */
    NLSSettings();

    /*!
     * Default settings initialization
     * \param settings the default settings
     */
    static void defaultSettings(NLSSettings& settings);

public:
    double noise_mean;
    double noise_std;
    double pos0_mean;
    double pos0_std;
    double reward_reg;

    virtual void WriteToStream(std::ostream& out) const;
    virtual void ReadFromStream(std::istream& in);

    virtual ~NLSSettings();
};

/*!
 * This class implements the NLS problem.
 * This problem is a two-dimensional MDP
 * where the aim is to let a robot reach
 * a goal state.
 *
 * References
 * ==========
 * [Vlassis, Toussaint, Kontes, Piperidis. Learning Model-free Robot Control by a Monte Carlo EM Algorithm](http://www.robolab.tuc.gr/ASSETS/PAPERS_PDF/PAPERS_2009/LEARNING_MODEL__EM_ALGOTITHM.pdf)
 */
class NLS: public ContinuousMDP
{
public:
    /*!
     * Constructor.
     */
    NLS();

    /*!
     * Constructor.
     * \param config the initial settings
     */
    NLS(NLSSettings& config);

    virtual ~NLS()
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
     * \see Environment::getSettings
     */
    inline const NLSSettings& getSettings() const
    {
        return *config;
    }

private:
    NLSSettings* config;
    bool cleanConfig;
};

}

#endif /* NLS_H_ */
