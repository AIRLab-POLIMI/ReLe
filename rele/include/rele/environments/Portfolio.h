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

#ifndef PORTFOLIO_H_
#define PORTFOLIO_H_

#include "rele/core/DenseMDP.h"

#define T_STEPS 50
#define N_STEPS 4
#define ALPHA 0.2
#define P_RISK 0.05
#define P_SWITCH 0.1
#define RL 1.0
#define RNL_HIGH 2
#define RNL_LOW 1.1

namespace ReLe
{

/*!
 * This class contains the settings of the Portfolio problem
 * and some functions to manage them.
 */
class PortfolioSettings : public EnvironmentSettings
{
public:
    /*!
     * Constructor.
     */
    PortfolioSettings();

    /*!
     * Default settings initialization
     * \param settings the default settings
     */
    static void defaultSettings(PortfolioSettings& settings);

    virtual ~PortfolioSettings()
    {
    }

public:
    // time dependent variables
    unsigned int t;
    double rNL;
    double T_rNL;
    double retL;
    double retNL;
    double T_Ret_Inv;

    // time independent variables
    unsigned int T;
    unsigned int n;
    double alpha;
    double P_Risk;
    double P_Switch;
    double rL;
    double rNL_High;
    double rNL_Low;
    double initial_state;

    virtual void WriteToStream(std::ostream& out) const;
    virtual void ReadFromStream(std::istream& in);
};

/*!
 * This class implements the Portfolio problem.
 * The aim of this problem is to maximize the expected
 * utility of a Portfolio in a financial market.
 * For further information see <a href="http://www.math.kit.edu/stoch/~baeuerle/seite/markov_nb/media/br_mdpjump.pdf">here</a>.
 *
 * References
 * ==========
 * [Di Castro, Tamar, Mannor. Policy gradients with variance related risk criteria. ICML 2012](http://machinelearning.wustl.edu/mlpapers/paper_files/ICML2012Di_489.pdf)
 */
class Portfolio: public DenseMDP
{
public:
    /*!
     * Constructor.
     */
    Portfolio();

    /*!
     * Constructor.
     * \param config the initial settings
     */
    Portfolio(PortfolioSettings& config);

    virtual ~Portfolio()
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
    inline const PortfolioSettings& getSettings() const
    {
        return *config;
    }

private:
    void defaultValues();
    PortfolioSettings* config;
    bool cleanConfig;

};

}

#endif /* PORTFOLIO_H_ */
