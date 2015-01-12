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

#include "td/SARSA.h"

namespace ReLe
{

SARSA_lambda::SARSA_lambda(double lambda) :
			lambda(lambda)
{

}

void SARSA_lambda::initEpisode()
{

}

void SARSA_lambda::sampleAction(const FiniteState& state, FiniteAction& action)
{

}

void SARSA_lambda::step(const Reward& reward, const FiniteState& nextState)
{

}

void SARSA_lambda::endEpisode(const Reward& reward)
{

}

SARSA_lambda::~SARSA_lambda()
{

}

}
