/*
 * rele,
 *
 *
 * Copyright (C) 2015 Matteo Pirotta & Davide Tateo
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

#ifndef INCLUDE_ALGORITHMS_TD_LINEARSARSA_H_
#define INCLUDE_ALGORITHMS_TD_LINEARSARSA_H_

#include "td/TD.h"
#include <armadillo>

namespace ReLe
{

/**
 * http://jmlr.org/proceedings/papers/v32/seijen14.pdf
 */
class LinearGradientSARSA: public LinearTD
{
public:
    LinearGradientSARSA(ActionValuePolicy<DenseState>& policy, Features& phi);
    virtual void initEpisode(const DenseState& state, FiniteAction& action) override;
    virtual void sampleAction(const DenseState& state, FiniteAction& action) override;
    virtual void step(const Reward& reward, const DenseState& nextState,
                      FiniteAction& action) override;
    virtual void endEpisode(const Reward& reward) override;

    virtual ~LinearGradientSARSA();

    void setReplacingTraces(bool val)
    {
        useReplacingTraces = val;
    }

private:
    arma::vec eligibility;
    double lambda;
    bool useReplacingTraces;

};

}

#endif /* INCLUDE_ALGORITHMS_TD_LINEARSARSA_H_ */
