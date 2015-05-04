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

#ifndef SRC_TEST_ROCKY_ROCKYOPTIONS_H_
#define SRC_TEST_ROCKY_ROCKYOPTIONS_H_

#include "Options.h"

namespace ReLe
{


class RockyOption : public FixedOption<DenseAction, DenseState>
{
public:
	RockyOption();

protected:
	arma::vec wayPointPolicy(const arma::vec& state, double ox, double oy);
	double angularDistance(const arma::vec& state, double ox, double oy);
	bool objectiveFree(const arma::vec& state, double ox, double oy);
	double rockyRelRotation(const arma::vec& state);

protected:
    enum StateComponents
    {
		//robot state
		x = 0,
		y,
		theta,

		//robot sensors
		energy,
		food,

		//rocky state
		xr,
		yr,
		thetar
    };

    double dt;
    const double maxV;

};

class Eat : public RockyOption
{
public:
	 virtual bool canStart(const DenseState& state);
	 virtual double terminationProbability(const DenseState& state);
	 virtual void operator ()(const DenseState& state, DenseAction& action);
};

class Home : public RockyOption
{
public:
	 virtual bool canStart(const DenseState& state);
	 virtual double terminationProbability(const DenseState& state);
	 virtual void operator ()(const DenseState& state, DenseAction& action);
};

class Feed : public RockyOption
{
public:
	Feed();
	 virtual bool canStart(const DenseState& state);
	 virtual double terminationProbability(const DenseState& state);
	 virtual void operator ()(const DenseState& state, DenseAction& action);

private:
	 arma::vec spot;
};

class Escape : public RockyOption
{
public:
	 virtual bool canStart(const DenseState& state);
	 virtual double terminationProbability(const DenseState& state);
	 virtual void operator ()(const DenseState& state, DenseAction& action);
};

}

#endif /* SRC_TEST_ROCKY_ROCKYOPTIONS_H_ */
