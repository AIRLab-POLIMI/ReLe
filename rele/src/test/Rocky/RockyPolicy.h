/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo
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

#ifndef SRC_TEST_ROCKY_ROCKYPOLICY_H_
#define SRC_TEST_ROCKY_ROCKYPOLICY_H_

namespace ReLe
{

class RockyPolicy : public ParametricPolicy<DenseAction, DenseState>
{
public:

	RockyPolicy();

    //Policy
    virtual arma::vec operator() (const arma::vec& state);
    virtual double operator() (const arma::vec& state, const arma::vec& action);

    inline virtual std::string getPolicyName()
    {
        return "Rocky Policy";
    }

    inline virtual std::string getPolicyHyperparameters()
    {
        return "";
    }

    inline virtual std::string printPolicy()
    {
        return "";
    }

    //ParametricPolicy
    inline virtual const arma::vec& getParameters() const
    {
        return w;
    }

    inline virtual const unsigned int getParametersSize() const
    {
        return w.n_elem;
    }

    virtual void setParameters(arma::vec& w)
    {
    	this->w = w;
    }


private:
    enum Objective
	{
    	escape, feed, eat, home
	};

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

    enum ParameterComponents
    {
    	escapeThreshold = 0,
		homeThreshold
    };


private:
    Objective computeObjective(const arma::vec& state);
    arma::vec eatPolicy();


private:
    arma::vec w;
};

}


#endif /* SRC_TEST_ROCKY_ROCKYPOLICY_H_ */
