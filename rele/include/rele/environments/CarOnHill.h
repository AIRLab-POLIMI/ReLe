/*
 * rele,
 *
 *
 * Copyright (C) 2016 Davide Tateo
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

#ifndef INCLUDE_RELE_ENVIRONMENTS_CARONHILL_H_
#define INCLUDE_RELE_ENVIRONMENTS_CARONHILL_H_

#include "rele/utils/ArmadilloOdeint.h"
#include "rele/core/DenseMDP.h"
#include <boost/numeric/odeint.hpp>

namespace ReLe
{

class CarOnHillSettings : public EnvironmentSettings
{
public:
    /*!
     * Constructor.
     */
	CarOnHillSettings();

    /*!
     * Default settings initialization
     * \param settings the default settings
     */
    static void defaultSettings(CarOnHillSettings& settings);

    virtual ~CarOnHillSettings();
    virtual void WriteToStream(std::ostream& out) const;
    virtual void ReadFromStream(std::istream& in);

public:
    double h;
    double m;

    double dt;
};

/*!
 * This class implements the Car On Hill problem.
 * This is a version of mountain car environment, the one proposed by Ernst paper, and is simpler than
 * the original mountain car problem, as the goal can be reached by a random policy.
 *
 * \see MountainCar
 *
 * References
 * ==========
 * [ErnstT, Geurts and Wehrnkel. Tree-Based Batch Mode Reinforcement Learning](http://www.jmlr.org/papers/volume6/ernst05a/ernst05a.pdf)
 */
class CarOnHill: public DenseMDP
{
	typedef arma::vec state_type;

private:
    //used in odeint
    class CarOnHillOde
    {

    public:
    	double action;

        CarOnHillOde(CarOnHillSettings& config);

        void operator()(const state_type& x, state_type& dx,
                        const double /* t */);

    private:
        double m;

        static constexpr double g = 9.81;
    };

public:
    enum StateLabel
    {
        position = 0, velocity = 1
    };

public:
    /*!
     * Constructor.
     */
    CarOnHill();

    /*!
     * Constructor.
     * \param config the initial settings
     */
    CarOnHill(CarOnHillSettings& config);

    /*!
     * Destructor.
     */
    virtual ~CarOnHill()
    {
        if(cleanConfig)
        	delete carOnHillConfig;
    }

    /*!
     * \see Environment::step
     */
    virtual void step(const FiniteAction& action, DenseState& nextState,
                      Reward& reward) override;

    /*!
     * \see Environment::getInitialState
     */
    virtual void getInitialState(DenseState& state) override;

    /*!
     * \see Environment::getSettings
     */
    inline const CarOnHillSettings& getSettings() const
    {
        return *carOnHillConfig;
    }

private:
    CarOnHillSettings* carOnHillConfig;
    CarOnHillOde carOnHillOde;
    bool cleanConfig;

    //[ define_adapt_stepper
    typedef boost::numeric::odeint::runge_kutta_cash_karp54< state_type > error_stepper_type;
    typedef boost::numeric::odeint::controlled_runge_kutta< error_stepper_type > controlled_stepper_type;
    controlled_stepper_type controlled_stepper;
};

}

#endif /* INCLUDE_RELE_ENVIRONMENTS_CARONHILL_H_ */
