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

#ifndef INCLUDE_RELE_ENVIRONMENTS_UNICYCLE_H_
#define INCLUDE_RELE_ENVIRONMENTS_UNICYCLE_H_

#include <boost/numeric/odeint.hpp>
#include "rele/core/ContinuousMDP.h"
#include "rele/utils/ArmadilloOdeint.h"
#include "rele/policy/Policy.h"

namespace ReLe
{

/*!
 * This class contains the settings of the Unicycle problem
 * and some functions to manage them.
 */
class UnicyclePolarSettings : public EnvironmentSettings
{
public:
    /*!
     * Constructor.
     */
    UnicyclePolarSettings();

    /*!
     * Default settings initialization
     * \param settings the default settings
     */
    static void defaultSettings(UnicyclePolarSettings& settings);

    virtual ~UnicyclePolarSettings();

public:
    double dt;
    double reward_th;

    virtual void WriteToStream(std::ostream& out) const;
    virtual void ReadFromStream(std::istream& in);
};

/*!
 * This class implements the Unicycle problem.
 * The aim of this problem is to control a unicycle
 * in order to let it stay balanced.
 *
 * \f[
 * \hat{e}(t) = [x(t) - x_g; y(t) - y_g; \theta(t) - \theta_g]^{T}
 * \f]
 * \f[
 * e(t) = \begin{bmatrix} e_x(t)\\e_y(t)\\ e_\theta(t)\end{bmatrix}
 *      = \begin{bmatrix}  cos\theta_g & sin\theta_g & 0\\
 *                        -sin\theta_g & cos\theta_g & 0\\
 *                         0 & 0 & 1
 *        \end{bmatrix} \hat{e}(t)
 * \f]
 * \f[
 * \rho   = \sqrt(e_x^2+e_y^2)
 * \f]
 * \f[
 * \gamma = atan2(e_y,e_x) - e_\theta + \pi
 * \f]
 * \f[
 * \delta = \gamma + e_\theta
 * \f]
 *
 * Optimal control law:
 * \f[
 * v = k_1 \rho cos\gamma
 * \f]
 * \f[
 * w = k_2 \gamma + k_1 \sin\gamma \cos\gamma (\gamma + k_3 \delta) / \gamma
 * \f]
 *
 * References
 * ==========
 * [Master Thesis](http://www.diva-portal.se/smash/get/diva2:662268/FULLTEXT01.pdf)
 *
 * [Stabilized Feedback Control of Unicycle Mobile Robots](http://cdn.intechopen.com/pdfs-wm/44029.pdf)
 */
class UnicyclePolar: public ContinuousMDP
{
public:
    enum StateLabel
    {
        rho = 0, gamma = 1, delta = 2
    };

    enum ActionLabel
    {
        linearVel = 0, angularVel = 1
    };


    typedef arma::vec state_type;

private:
    //used in odeint
    class UnicyclePolarOde
    {

    public:
        double v, w;

        void operator()(const state_type& x, state_type& dx,
                        const double /* t */);

    };

public:
    /*!
     * Constructor.
     */
    UnicyclePolar();

    /*!
     * Constructor.
     * \param config the initial settings
     */
    UnicyclePolar(UnicyclePolarSettings& config);

    virtual ~UnicyclePolar()
    {
        if (cleanConfig)
            delete unicycleConfig;
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
    inline const UnicyclePolarSettings& getSettings() const
    {
        return *unicycleConfig;
    }

private:
    UnicyclePolarSettings* unicycleConfig;
    UnicyclePolarOde unicycleode;
    bool cleanConfig;

    //[ define_adapt_stepper
    typedef boost::numeric::odeint::runge_kutta_cash_karp54< state_type > error_stepper_type;
    typedef boost::numeric::odeint::controlled_runge_kutta< error_stepper_type > controlled_stepper_type;
    controlled_stepper_type controlled_stepper;
    //]

};


//=================================================
// Unicycle policy
//-------------------------------------------------
class UnicycleControlLaw : public ParametricPolicy<DenseAction, DenseState>
{
public:
    UnicycleControlLaw()
        : params(3, arma::fill::zeros)
    {
    }
    UnicycleControlLaw(arma::vec p)
        : params(p)
    {
        assert(p.n_elem == 3);
    }

    // Policy interface
public:
    arma::vec operator() (const arma::vec& state) override;
    double operator() (const arma::vec& state, const arma::vec& action) override;

    inline std::string getPolicyName() override
    {
        return "UnicycleControlLaw";
    }

    inline std::string printPolicy() override
    {
        return " ";
    }

    Policy<DenseAction, DenseState>* clone() override;

    // ParametricPolicy interface
public:
    arma::vec getParameters() const override;
    const unsigned int getParametersSize() const override;
    void setParameters(const arma::vec& w) override;

protected:
    arma::vec params;
};

}

#endif /* INCLUDE_RELE_ENVIRONMENTS_UNICYCLE_H_ */
