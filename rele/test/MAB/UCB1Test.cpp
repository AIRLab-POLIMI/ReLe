/*
 * MABTest.cpp
 *
 *  Created on: Dec 22, 2015
 *      Author: francesco
 */

#include "rele/environments/MAB/InternetAds.h"
#include "rele/algorithms/td/DoubleQ-Learning.h"
#include "rele/policy/parametric/IdentityPolicy.h"
#include "rele/core/Core.h"
#include "rele/algorithms/MABAlgorithm/UCB1.h"

#include <iostream>

using namespace std;
using namespace ReLe;


/*
 * MAB test with InternetAds Sequential policy is used
 * with the purpose to execute each possible action sequentially until
 * the end of the episode.
 */

int main(int argc, char *argv[])
{
    unsigned int nAds = 10;
    unsigned int episodeLength = 1;

    InternetAds mab(nAds, InternetAds::Second);

    IdentityPolicy<FiniteState> policy;
    UCB1<FiniteState> agent(policy);

    auto&& core = buildCore(mab, agent);
    core.getSettings().episodeLength = 1;
    /*for(unsigned int i = 0; i < mab.getVisitors(); i++)
    {
    	cout << endl << "### Starting episode " << i << " ###" << endl;
    	core.runEpisode();

    	arma::vec P = mab.getP();
    	cout << endl << "P: " << P << endl;
    }*/
}




