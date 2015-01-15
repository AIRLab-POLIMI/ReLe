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
#include "Logger.h"

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
				Agent<ActionC, StateC>& agent) :
				envirorment(envirorment), agent(agent)
	{

	}

	CoreSettings& getSettings()
	{
		return settings;
	}

	void runEpisode()
	{
		Logger<ActionC, StateC> logger;
		StateC xn;
		ActionC u;

		envirorment.getInitialState(xn);
		agent.initEpisode(xn, u);
		logger.log(xn);

		for (unsigned int i = 0; i < settings.episodeLenght; i++)
		{
			Reward r;

			envirorment.step(u, xn, r);

			if(xn.isAbsorbing())
			{
				agent.endEpisode(r);
				logger.log(xn, r);
				break;
			}

			agent.step(r, xn, u);
			logger.log(u, xn, r, i);
		}

		if(!xn.isAbsorbing())
			agent.endEpisode();

		logger.printStatistics();
	}

	void setupAgent()
	{
		const EnvirormentSettings& task = envirorment.getSettings();

		if (!task.isFiniteHorizon)
		{
			agent.setGamma(task.gamma);
		}
	}

private:
	Envirorment<ActionC, StateC>& envirorment;
	Agent<ActionC, StateC>& agent;
	CoreSettings settings;
};

}

#endif /* CORE_H_ */
