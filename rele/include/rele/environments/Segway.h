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

#ifndef INCLUDE_RELE_ENVIRONMENTS_SEGWAY_H_
#define INCLUDE_RELE_ENVIRONMENTS_SEGWAY_H_

#include "rele/utils/ArmadilloOdeint.h"
#include "rele/core/ContinuousMDP.h"
#include <boost/numeric/odeint.hpp>

namespace ReLe
{

/*!
 * This class contains the settings of the Segway problem
 * and some functions to manage them.
 */
class SegwaySettings : public EnvironmentSettings
{
public:
    /*!
     * Constructor.
     */
    SegwaySettings();

    /*!
     * Default settings initialization
     * \param settings the default settings
     */
    static void defaultSettings(SegwaySettings& settings);

    virtual ~SegwaySettings();
    virtual void WriteToStream(std::ostream& out) const;
    virtual void ReadFromStream(std::istream& in);

public:
    double l;
    double r;
    double Ir;
    double Ip;
    double Mp;
    double Mr;

    double dt;
};

/*!
 * This class implements a continuous MDP problem where
 * a segway has to be controlled in order to balance it.
 *
 * References
 * ==========
 * [Tesi](http://repository.tudelft.nl/assets/uuid:6e9714e4-2057-444c-a184-9401242cf1a8/Thesis_Xueli_Jia.pdf).
 */
class Segway: public ContinuousMDP
{
    typedef arma::vec state_type;

private:
    //used in odeint
    class SegwayOde
    {

    public:
        double action;

        SegwayOde(SegwaySettings& config);

        void operator()(const state_type& x, state_type& dx,
                        const double /* t */);

    private:
        double l;
        double r;
        double Ir;
        double Ip;
        double Mp;
        double Mr;

        static constexpr double g = 9.81;

    };

public:
    /*!
     * Constructor.
     */
    Segway();

    /*!
     * Constructor.
     * \param config the initial settings
     */
    Segway(SegwaySettings& config);

    virtual ~Segway()
    {
        if (cleanConfig)
            delete segwayConfig;
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
    inline const SegwaySettings& getSettings() const
    {
        return *segwayConfig;
    }


private:
    SegwaySettings* segwayConfig;
    SegwayOde segwayode;
    bool cleanConfig;

    //[ define_adapt_stepper
    typedef boost::numeric::odeint::runge_kutta_cash_karp54< state_type > error_stepper_type;
    typedef boost::numeric::odeint::controlled_runge_kutta< error_stepper_type > controlled_stepper_type;
    controlled_stepper_type controlled_stepper;
    //]

};

}

#endif /* INCLUDE_RELE_ENVIRONMENTS_SEGWAY_H_ */
