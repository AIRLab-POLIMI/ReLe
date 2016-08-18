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

#ifndef INCLUDE_RELE_ENVIRONMENTS_ACROBOT_H_
#define INCLUDE_RELE_ENVIRONMENTS_ACROBOT_H_

#include "rele/utils/ArmadilloOdeint.h"
#include "rele/utils/ArmadilloPDFs.h"
#include "rele/core/DenseMDP.h"
#include "rele/utils/RandomGenerator.h"
#include <boost/numeric/odeint.hpp>

namespace ReLe
{

class AcrobotSettings : public EnvironmentSettings
{
public:
    /*!
     * Constructor.
     */
    AcrobotSettings();

    /*!
     * Default settings initialization
     * \param settings the default settings
     */
    static void defaultSettings(AcrobotSettings& settings);

    virtual ~AcrobotSettings();
    virtual void WriteToStream(std::ostream& out) const;
    virtual void ReadFromStream(std::istream& in);

public:
    double M1, M2;
    double L1, L2;
    double mu1, mu2;

    double dt;
};

/*!
 * This class implements the Acrobot problem.
 * This is the version of Acrobot environment proposed in Ernst paper.
 *
 * \see Acrobot
 *
 * References
 * ==========
 * [ErnstT, Geurts and Wehrnkel. Tree-Based Batch Mode Reinforcement Learning](http://www.jmlr.org/papers/volume6/ernst05a/ernst05a.pdf)
 */
class Acrobot: public DenseMDP
{
    typedef arma::vec state_type;

private:
    //used in odeint
    class AcrobotOde
    {

    public:
        double action;

        AcrobotOde(AcrobotSettings& config);

        void operator()(const state_type& x, state_type& dx,
                        const double /* t */);

    private:
        double M1, M2;
        double L1, L2;
        double mu1, mu2;

        static constexpr double g = 9.81;
    };

public:
    enum StateLabel
    {
        theta1idx = 0, theta2idx = 1, dTheta1idx = 2, dTheta2idx = 3
    };

public:
    /*!
     * Constructor.
     */
    Acrobot();

    /*!
     * Constructor.
     * \param config the initial settings
     */
    Acrobot(AcrobotSettings& config);

    /*!
     * Destructor.
     */
    virtual ~Acrobot()
    {
        if(cleanConfig)
            delete acrobotConfig;
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
    inline const AcrobotSettings& getSettings() const
    {
        return *acrobotConfig;
    }

private:
    AcrobotSettings* acrobotConfig;
    AcrobotOde acrobotOde;
    bool cleanConfig;
    RandomGenerator rg;

    //[ define_adapt_stepper
    typedef boost::numeric::odeint::runge_kutta_cash_karp54< state_type > error_stepper_type;
    typedef boost::numeric::odeint::controlled_runge_kutta< error_stepper_type > controlled_stepper_type;
    controlled_stepper_type controlled_stepper;
};

}

#endif /* INCLUDE_RELE_ENVIRONMENTS_ACROBOT_H_ */
