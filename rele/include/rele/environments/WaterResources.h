/*
 * rele,
 *
 *
 * Copyright (C) 2017 Davide Tateo
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

#ifndef INCLUDE_RELE_ENVIRONMENTS_WATERRESOURCES_H_
#define INCLUDE_RELE_ENVIRONMENTS_WATERRESOURCES_H_

#include "rele/core/ContinuousMDP.h"

namespace ReLe
{

class WaterResourcesSettings : public EnvironmentSettings
{
public:

    WaterResourcesSettings();
    static void defaultSettings(WaterResourcesSettings& settings);

    virtual ~WaterResourcesSettings();

public:
    //Noise parameters
    arma::vec mu;
    arma::mat Sigma;

    //Reservoirs parameters
    arma::vec maxCapacity;
    arma::vec S;

    //Flooding related
    arma::vec h_flo;

    //Irrigation parameters
    double w_irr;

    //Power related
    double w_hyd;
    double q_mef;
    double eta;
    double gamma_h20;

    double delta;

    virtual void WriteToStream(std::ostream& out) const;
    virtual void ReadFromStream(std::istream& in);
};

class WaterResources: public ContinuousMDP
{
public:
    /*!
     * Constructor.
     */
    WaterResources();

    /*!
     * Constructor.
     * \param config the initial settings
     */
    WaterResources(WaterResourcesSettings& config);

    /*!
     * \see Environment::step
     */
    virtual void step(const DenseAction& action, DenseState& nextState,
                      Reward& reward) override;

    /*!
     * \see Environment::getInitialState
     */
    virtual void getInitialState(DenseState& state) override;

    virtual ~WaterResources();

    enum StateComponents
    {
        // robot state
        up = 0,
        dn,
        // state size
        STATESIZE
    };

    enum RewardComponents
    {
        // robot state
        flo_up = 0,
        flo_dn,
        hyd,
        irr,
        // state size
        REWARDSIZE
    };

private:
    void computeReward(Reward& reward, const arma::vec& r);
    double computePowerGeneration(double h, double r);

private:
    WaterResourcesSettings* config;
    bool cleanConfig;
};


}

#endif /* INCLUDE_RELE_ENVIRONMENTS_WATERRESOURCES_H_ */
