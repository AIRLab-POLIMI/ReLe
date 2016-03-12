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

#ifndef MULTIHEAT_H_
#define MULTIHEAT_H_

#include "rele/core/DenseMDP.h"

namespace ReLe
{

/*!
 * This class contains the settings of the Multi Heat problem
 * and some functions to manage it.
 */
class MultiHeatSettings : public EnvironmentSettings
{
public:
    /*!
     * Constructor.
     */
    MultiHeatSettings();

    /*!
     * Default settings initialization
     * \param settings the default settings
     */
    static void defaultSettings(MultiHeatSettings& settings);

    virtual ~MultiHeatSettings();

    virtual void WriteToStream(std::ostream& out) const;
    virtual void ReadFromStream(std::istream& in);

public:
    unsigned int Nr;
    double Ta;
    double dt;
    double a;
    double s2n;

    arma::mat A;
    arma::vec B, C;

    double TUB, TLB;
};

/*!
 * This class implements the Multi Heat problem.
 * The aim of this problem is to find the optimal
 * heating policy according to environmental
 * conditions and other criterion of optimality.
 *
 * References
 * ==========
 * [Pirotta, Manganini, Piroddi, Prandini, Restelli. A particle-based policy for the optimal
control of Markov decision processes. CDC, 2014.](http://www.nt.ntnu.no/users/skoge/prost/proceedings/ifac2014/media/files/1987.pdf)
 */
class MultiHeat: public DenseMDP
{
public:
    MultiHeatSettings* config;
    bool cleanConfig;

    arma::mat Xi;
    arma::vec Gamma;

public:
    /*!
     * Constructor.
     */
    MultiHeat();

    /*!
     * Constructor.
     * \param config the initial settings
     */
    MultiHeat(MultiHeatSettings& config);

    virtual ~MultiHeat()
    {
        if (cleanConfig)
            delete config;
    }

    /*!
     * \see Environment::step
     */
    virtual void step(const FiniteAction& action, DenseState& nextState, Reward& reward) override;

    /*!
     * \see Environment::getInitialState
     */
    virtual void getInitialState(DenseState& state) override;

    /*!
     * Set the current state.
     * \param state the current state
     */
    void setCurrentState(DenseState& state) //TO REMOVE
    {
        currentState = state;
    }

private:
    unsigned int const mode = 0;

    void computeTransitionMatrix();
};

}

#endif /* MULTIHEAT_H_ */
