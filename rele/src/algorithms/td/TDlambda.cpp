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

#include "td/TDlambda.h"

namespace ReLe
{

TD_lambda::TD_lambda(double lambda) :
			lambda(lambda)
{

}

void TD_lambda::initEpisode()
{

}

void TD_lambda::sampleAction(const FiniteState& state, FiniteAction& action)
{

}

void TD_lambda::step(const Reward& reward, const FiniteState& nextState)
{

}

void TD_lambda::endEpisode(const Reward& reward)
{

}

TD_lambda::~TD_lambda()
{

}

}
