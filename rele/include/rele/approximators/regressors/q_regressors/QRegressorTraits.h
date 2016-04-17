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

#ifndef INCLUDE_RELE_APPROXIMATORS_REGRESSORS_Q_REGRESSORS_QREGRESSORTRAITS_H_
#define INCLUDE_RELE_APPROXIMATORS_REGRESSORS_Q_REGRESSORS_QREGRESSORTRAITS_H_

#include "rele/approximators/Regressors.h"
#include "rele/approximators/regressors/q_regressors/QRegressor.h"

namespace ReLe
{

/*
 * Helper trait
 */
template<class RegressorC>
struct q_regressor_trait
{
    static const bool isParametric = std::is_base_of<ParametricRegressor_<typename RegressorC::InputType, RegressorC::isDense>, RegressorC>::value;
    static const bool isSupervised = std::is_base_of<BatchRegressor_<typename RegressorC::InputType, arma::vec, RegressorC::isDense>, RegressorC>::value;
    static const bool isUnsupervised = std::is_base_of<UnsupervisedBatchRegressor_<typename RegressorC::InputType, arma::vec, RegressorC::isDense>, RegressorC>::value;
    static const bool isSimple = !(isParametric || isSupervised || isUnsupervised);
};


/*
 * Default traits
 */
template<bool isSimple>
struct q_regressor_simple
{

};

template<class RegressorC, bool isParametric>
struct q_regressor_parametric
{
	q_regressor_parametric(std::vector<RegressorC*>& regressors)
	{

	}

};

template<class RegressorC, bool isSupervised>
struct q_regressor_supervised
{
	q_regressor_supervised(std::vector<RegressorC*>& regressors)
	{

	}
};

/*template<bool isUnsupervised>
struct q_regressor_unsupervised
{

};*/


/*
 * Implementation traits
 */

template<>
struct q_regressor_simple<true> : public QRegressor
{

};

template<class RegressorC>
struct q_regressor_parametric<RegressorC, true> : public ParametricQRegressor
{
	q_regressor_parametric(std::vector<RegressorC*>& regressors)
				: regressors(regressors)
	{

	}

	virtual void set(unsigned int action, const arma::vec& w) override
	{
		auto& regressor = *regressors[action];
		regressor.setParameters(w);
	}

	virtual void update(unsigned int action, const arma::vec& dw) override
	{
		//TODO [INTERFACE] add update method to regressors
		auto& regressor = *regressors[action];
		arma::vec wOld = regressor.getParameters();
		regressor.setParameters(wOld+dw);
	}

	virtual arma::vec diff(const arma::vec state, unsigned int action) override
	{
		auto& regressor = *regressors[action];
		return regressor.diff(state);
	}

	virtual ~q_regressor_parametric()
	{

	}

	std::vector<RegressorC*>& regressors;
};

template<class RegressorC>
struct q_regressor_supervised<RegressorC, true> : public BatchQRegressor
{
	q_regressor_supervised(std::vector<RegressorC*>& regressors)
	{

	}

	virtual void trainFeatures() override
	{
		//TODO [IMPRTANT][INTERFACE] Implement & fix interface
	}
};

/*template<bool denseOutput>
struct q_regressor_unsupervised<true, denseOutput>: public Un
{

};*/

template<class RegressorC>
using simple_q_reg = q_regressor_simple<q_regressor_trait<RegressorC>::isSimple>;

template<class RegressorC>
using parametric_q_reg = q_regressor_parametric<RegressorC, q_regressor_trait<RegressorC>::isParametric>;

template<class RegressorC>
using supervised_q_reg = q_regressor_supervised<RegressorC, q_regressor_trait<RegressorC>::isSupervised>;


}



#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_Q_REGRESSORS_QREGRESSORTRAITS_H_ */
