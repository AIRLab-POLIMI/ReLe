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

#ifndef UNDERWATERVEHICLE_H_
#define UNDERWATERVEHICLE_H_

#include <boost/numeric/odeint.hpp>
#include "rele/core/DenseMDP.h"
#include "rele/utils/Range.h"
#include "rele/utils/ArmadilloOdeint.h"

namespace ReLe
{

/*!
 * This class contains the settings of the Underwater Vehicle problem
 * and some functions to manage them.
 */
class UWVSettings : public EnvironmentSettings
{
public:
    /*!
     * Constructor.
     */
    UWVSettings();

    /*!
     * Default settings initialization
     * \param settings the default settings
     */
    static void defaultSettings(UWVSettings& settings);

    virtual ~UWVSettings();

public:
    Range thrustRange;
    Range velocityRange;
    double dt;
    double C;
    double mu; //rad
    double setPoint; // m/s
    int timeSteps;
    std::vector<double> actionList;

    virtual void WriteToStream(std::ostream& out) const;
    virtual void ReadFromStream(std::istream& in);
};

/*!
 * This class implements the Underwater Vehicle problem.
 * The task of this problem is to control the speed an
 * underwater vehicle in a submarine environment modeling
 * the complex dynamics of objects moving in fluids.
 *
 * References
 * ==========
 * [Hafner, Riedmiller. Reinforcement learning in feedback control. Challenges and benchmarks from technical process control. Machine Learning](http://link.springer.com/article/10.1007%2Fs10994-011-5235-x)
 */
class UnderwaterVehicle: public DenseMDP
{
    typedef arma::vec state_type;

private:
    //used in odeint
    class UWVOde
    {

    public:
        double action;

        UWVOde() : action(0) { }

        void operator() ( const state_type& x , state_type& dxdt , const double /* t */ )
        {
            const double u     = action;
            const double v     = x[0];
            const double abs_v = fabs(v);
            const double c_v   = 1.2f + 0.2f * sin(abs_v);
            const double m_v   = 3.0f + 1.5f * sin(abs_v);
            const double k_v_u = -0.5f * tanh((fabs(c_v * v * abs_v - u) - 30.0f) * 0.1f) + 0.5f;
            dxdt[0] = (u * k_v_u - c_v * v * abs_v) / m_v;
        }
    };

public:
    /*!
     * Constructor.
     */
    UnderwaterVehicle();

    /*!
     * Constructor.
     * \param config the initial settings
     */
    UnderwaterVehicle(UWVSettings& config);

    virtual ~UnderwaterVehicle()
    {
        if (cleanConfig)
            delete config;
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
    inline const UWVSettings& getSettings() const
    {
        return *config;
    }

private:
    UWVSettings* config;
    UWVOde uwvode;
    bool cleanConfig;

    //[ define_adapt_stepper
    typedef boost::numeric::odeint::runge_kutta_cash_karp54< state_type > error_stepper_type;
    typedef boost::numeric::odeint::controlled_runge_kutta< error_stepper_type > controlled_stepper_type;
    controlled_stepper_type controlled_stepper;
    //]
};

}

#endif /* UNDERWATERVEHICLE_H_ */
