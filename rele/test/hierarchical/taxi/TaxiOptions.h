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

#include "rele/policy/Options.h"

namespace ReLe
{

class TaxiLocationOption : public FixedOption<FiniteAction, DenseState>
{
public:
    TaxiLocationOption(arma::vec location);
    virtual bool canStart(const arma::vec& state) override;
    virtual double terminationProbability(const DenseState& state) override;
    virtual void operator ()(const DenseState& state, FiniteAction& action) override;

protected:
    void goToLocation(const DenseState& state, FiniteAction& action);

protected:
    arma::vec location;

};

class TaxiSimpleOption :  public FixedOption<FiniteAction, DenseState>
{
public:
    virtual bool canStart(const arma::vec& state) override;
    virtual double terminationProbability(const DenseState& state) override;
};

class TaxiPickupOption : public TaxiSimpleOption
{
public:
    virtual void operator ()(const DenseState& state, FiniteAction& action) override;

};

class TaxiDropOffOption : public TaxiSimpleOption
{
public:
    virtual void operator ()(const DenseState& state, FiniteAction& action) override;

};

class TaxiFillUpOption : public TaxiSimpleOption
{
public:
    virtual void operator ()(const DenseState& state, FiniteAction& action) override;

};

class TaxiComplexOption : public FixedOption<FiniteAction, DenseState>
{
protected:
    enum ActionType
    {
        DropOff, PickUp, FillUp
    };

public:
    TaxiComplexOption(std::vector<arma::vec2>& location, ActionType action);
    virtual bool canStart(const arma::vec& state) override;
    virtual void operator ()(const DenseState& state, FiniteAction& action) override;


protected:
    arma::vec& getLocation(const DenseState& state);
    void goToLocation(const DenseState& state, FiniteAction& action);

protected:
    std::vector<arma::vec2> locations;
    ActionType actionType;


};

class TaxiComplexPickupOption : public TaxiComplexOption
{
public:
    TaxiComplexPickupOption(std::vector<arma::vec2>& locations);
    virtual double terminationProbability(const DenseState& state) override;

};

class TaxiComplexDropOffOption : public TaxiComplexOption
{
public:
    TaxiComplexDropOffOption(std::vector<arma::vec2>& location);
    virtual double terminationProbability(const DenseState& state) override;

};

class TaxiComplexFillupOption : public TaxiComplexOption
{
public:
    TaxiComplexFillupOption(std::vector<arma::vec2>& location);
    virtual double terminationProbability(const DenseState& state) override;

};



}

#endif /* SRC_TEST_HIERARCHICAL_TAXI_TAXIOPTIONS_H_ */
