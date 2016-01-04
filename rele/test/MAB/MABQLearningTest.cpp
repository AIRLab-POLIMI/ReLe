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

/*
 * Written by: Carlo D'Eramo
 */

#include "MAB/InternetAds.h"
#include "td/DoubleQ-Learning.h"
#include "td/WQ-Learning.h"
#include "nonparametric/SequentialPolicy.h"
#include "Core.h"

#include <iostream>

using namespace std;
using namespace ReLe;


/*
 * MAB test with InternetAds environment and Q-Learning algorithm.
 * Sequential policy is used with the purpose to execute each possible
 * action sequentially until the end of the episode.
 */

int main(int argc, char *argv[])
{
    unsigned int nAds = 10;
    unsigned int episodeLength = 1;

    InternetAds mab(nAds, InternetAds::First);

    SequentialPolicy policy(mab.getSettings().finiteActionDim, episodeLength);
    //Q_Learning agent(policy);
    //DoubleQ_Learning agent(policy);
    WQ_Learning agent(policy);
    agent.setAlpha(0.005);

    auto&& core = buildCore(mab, agent);
    core.getSettings().episodeLength = episodeLength;
    for(unsigned int i = 0; i < mab.getVisitors(); i++)
    {
        cout << endl << "### Starting episode " << i << " ###" << endl;
        core.runEpisode();

        arma::vec P = mab.getP();
        cout << endl << "P: " << P << endl;
    }
}

