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

#ifndef CORE_H_
#define CORE_H_

#include "Envirorment.h"

//TODO: move all code in cpp and include at the bottom of the file
//TODO: or leave the code in this file

namespace ReLe
{

struct CoreSettings
{
	unsigned int episodeLenght;

};

template<class ActionC, class StateC>
class Core
{
	static_assert(std::is_base_of<Action, ActionC>::value, "Not valid Action class as template parameter");
	static_assert(std::is_base_of<State, StateC>::value, "Not a valid State class as template parameter");
public:
	Core(Envirorment<ActionC, StateC>& envirorment,
				Agent<ActionC, StateC>& state) :
				envirorment(envirorment), agent(agent)
	{

	}

	CoreSettings& getSettings()
	{
		return settings;
	}

	void runEpisode()
	{
		StateC xn;
		ActionC u;

		envirorment.getInitialState(xn);

		for (int i = 0; i < settings.episodeLenght && !xn.isAbsorbing(); i++)
		{
			Reward r;
			agent.sampleAction(xn, u);
			envirorment.step(u, xn, r);
			agent.step(r, xn);
		}
	}

	/*void setupAgent() serve?
	{
		EnvirormentSettings& task = envirorment.getSettings();

		if (!task.isFiniteHorizon)
		{
			//set gamma
		}
	}*/

private:
	Envirorment<ActionC, StateC>& envirorment;
	Agent<ActionC, StateC>& agent;
	CoreSettings settings;
};

}

#endif /* CORE_H_ */
