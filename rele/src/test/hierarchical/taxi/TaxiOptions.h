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

#ifndef SRC_TEST_HIERARCHICAL_TAXI_TAXIOPTIONS_H_
#define SRC_TEST_HIERARCHICAL_TAXI_TAXIOPTIONS_H_

#include "Options.h"

namespace ReLe
{

class TaxiLocationOption : public FixedOption<FiniteAction, DenseState>
{
public:
	TaxiLocationOption(arma::vec location);
	virtual bool canStart(const arma::vec& state);
	virtual double terminationProbability(const DenseState& state);

protected:
	void goToLocation(const DenseState& state, FiniteAction& action);

protected:
	arma::vec location;
	bool terminate;
};


class TaxiPickupOption : public TaxiLocationOption
{
public:
	TaxiPickupOption(arma::vec location);
	virtual void operator ()(const DenseState& state, FiniteAction& action);
};

class TaxiDropOffOption : public TaxiLocationOption
{
public:
	TaxiDropOffOption(arma::vec location);
	virtual void operator ()(const DenseState& state, FiniteAction& action);

};

class TaxiFillUpOption : public TaxiLocationOption
{
public:
	TaxiFillUpOption(arma::vec location);
	virtual void operator ()(const DenseState& state, FiniteAction& action);
	//TaxiFillUpOption(arma::vec location);


};


}

#endif /* SRC_TEST_HIERARCHICAL_TAXI_TAXIOPTIONS_H_ */
