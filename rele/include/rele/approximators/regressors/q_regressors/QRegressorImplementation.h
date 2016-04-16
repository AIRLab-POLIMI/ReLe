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

#ifndef INCLUDE_RELE_APPROXIMATORS_REGRESSORS_Q_REGRESSORS_QREGRESSORIMPLEMENTATION_H_
#define INCLUDE_RELE_APPROXIMATORS_REGRESSORS_Q_REGRESSORS_QREGRESSORIMPLEMENTATION_H_

#include "rele/approximators/regressors/q_regressors/QRegressorTraits.h"

namespace ReLe
{

template<class RegressorC>
class QRegressor_ : public simple_q_reg<RegressorC>,
    public parametric_q_reg<RegressorC>,
    public supervised_q_reg<RegressorC>

{
public:
    QRegressor_(const std::vector<RegressorC*>& regressors)
        : regressors(regressors)
    {

    }

    virtual double operator()(const arma::vec& state, unsigned int action) override
    {
    	auto& Q_a = *regressors[action];
    	return Q_a(state);
    }

    virtual ~QRegressor_()
    {

    }

private:
    std::vector<RegressorC*> regressors;
};

}

#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_Q_REGRESSORS_QREGRESSORIMPLEMENTATION_H_ */
