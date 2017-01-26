/*
 * rele,
 *
 *
 * Copyright (C) 2017 Davide Tateo
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

#include <rele/approximators/features/SparseFeatures.h>
#include <rele/approximators/features/DenseFeatures.h>

#include <rele/approximators/regressors/others/LinearApproximator.h>
#include <rele/approximators/basis/FrequencyBasis.h>
#include <rele/approximators/basis/HaarWavelets.h>

#include <rele/policy/parametric/differentiable/NormalPolicy.h>
#include <rele/IRL/algorithms/MLEDistributionLinear.h>
#include <rele/utils/ArmadilloExtensions.h>


#include <rele/utils/FileManager.h>

#include <rele/environments/EmptyEnv.h>
#include <rele/core/PolicyEvalAgent.h>
#include <rele/core/Core.h>

#include "rele_ros/bag/RosDataset.h"
#include "rele_ros/bag/message/RosGeometryInterface.h"

#include <boost/filesystem.hpp>

using namespace std;
using namespace arma;
using namespace boost::filesystem;
using namespace ReLe;
using namespace ReLe_ROS;

void preprocessDataset(Dataset<DenseAction, DenseState>& data)
{
	for(unsigned int ep = 0; ep < data.size(); ep++)
	{
		//truncate start
		auto& episode = data[ep];

		unsigned int i;
		for(i = 0; i < data[ep].size(); i++)
		{
			auto& tr = episode[i];

			if(tr.u(0) != 0 || tr.u(1) != 0 || tr.u(2) != 0)
			{
				break;
			}
		}

		episode.erase(episode.begin(), episode.begin() + i);

		//update timestamp
		double t0 = episode[0].x(0);

		for(i = 0; i < data[ep].size(); i++)
		{
			auto& tr = episode[i];

			tr.x(0) -= t0;
		}


		//truncate end of episode
		for(i = episode.size() - 1; i > 0; i--)
		{
			auto& tr = episode[i];

			if(tr.u(0) != 0 || tr.u(1) != 0 || tr.u(2) != 0)
			{
				break;
			}
		}

		episode.erase(episode.begin() + i + 1, episode.end());

		//set episode lenght to 4 seconds
		double tf = episode.back().x(0);
		double dt = tf - episode[episode.size() - 2].x(0);
		while(tf < 4.0)
		{
			tf += dt;
			Transition<DenseAction, DenseState> tr;

		    DenseAction u(3);
		    DenseState x(1), xn(1);

		    x(0) = tf;
		    xn(0) = tf + dt;
		    u(0) = 0;
		    u(1) = 0;
		    u(2) = 0;

			tr.x = x;
			tr.u = u;
			tr.xn = xn;
			episode.push_back(tr);
		}

	}
}

int main(int argc, char *argv[])
{
    FileManager fm("emotions", "arrabbiato");
    fm.createDir();
    //fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    //Read emotion datatset
    auto* t1 = new RosTopicInterface_<geometry_msgs::Twist>("/cmd_vel", true, true);
    std::vector<RosTopicInterface*> topics;
    topics.push_back(t1);

    RosDataset rosDataset(topics);

    std::string basePath = "/home/dave/Dropbox/Dottorato/Major/test/arrabbiato/";

    int count = 0;

    boost::filesystem::directory_iterator end_itr;
    for(boost::filesystem::directory_iterator i(basePath); i != end_itr; ++i )
    {
        if(boost::filesystem::is_regular_file(i->status()) &&
        			i->path().extension() == ".bag")
        {
        	cout << count++ << std::endl;
        	cout << i->path().string() << endl;
        	rosDataset.readEpisode(i->path().string());
        }
    }


    preprocessDataset(rosDataset.getData());

    //Create basis function for policy
    int uDim = 3;
    /*double maxT = rosDataset.getData()[0].back().x(0);

    unsigned int N = rosDataset.getData().getTransitionsNumber();
    double df = 1/maxT*2;
    double fE = 20.0;

    std::cout << "df: " << df << " fe: " << fE << " N: " << N << " tmax: "
    			<< maxT << " 1/tmax: " << 1.0/maxT << endl;

    BasisFunctions basis = FrequencyBasis::generate(0, df, fE, df, true);
   	BasisFunctions tmp = FrequencyBasis::generate(0, 0, fE, df, false);
    basis.insert(basis.end(), tmp.begin(), tmp.end());*/
    BasisFunctions basis = HaarWavelets::generate(0, 5, 4);

    SparseFeatures phi(basis, uDim);

    //Fit Normal distribution
    /*BasisFunctions basisEst = FrequencyBasis::generate(0, df, fE, df, true);
    BasisFunctions tmpEst = FrequencyBasis::generate(0, 0, fE, df, false);
    basisEst.insert(basisEst.end(), tmpEst.begin(), tmpEst.end());*/
    BasisFunctions basisEst = HaarWavelets::generate(0, 5, 4);
    DenseFeatures phiEst(basisEst);

    MLEDistributionLinear estimator(phiEst);
    estimator.compute(rosDataset.getData());

    auto theta = estimator.getParameters();

    arma::mat Cov = arma::cov(theta.t());
    auto M = safeChol(Cov);
    ParametricNormal dist(arma::mean(theta, 1), M*M.t());
    MVNPolicy policy(phi, arma::eye(uDim, uDim)*1e-3);


    /* Testing */
    EmptyEnv env(uDim, 100.0);

    CollectorStrategy<DenseAction, DenseState> f;

    //Compute fitted trajectory for each demonstration
    for(int i = 0; i < theta.n_cols; i++)
    {
    	policy.setParameters(theta.col(i));
        PolicyEvalAgent<DenseAction, DenseState> agent(policy);
        Core<DenseAction, DenseState> core(env, agent);

        core.getSettings().episodeLength = 2000;
        core.getSettings().loggerStrategy = &f;
        core.runTestEpisode();
    }

    //Test the fitted distribution
    PolicyEvalDistribution<DenseAction, DenseState> agent(dist, policy);
    Core<DenseAction, DenseState> core(env, agent);

    CollectorStrategy<DenseAction, DenseState> s;
    core.getSettings().episodeLength = 2000;
    core.getSettings().testEpisodeN = theta.n_cols;
    core.getSettings().loggerStrategy = &s;
    core.runTestEpisodes();


    /* Save the dataset in ReLe format */
    std::ofstream os1(fm.addPath("expert_dataset.log"));
    rosDataset.getData().writeToStream(os1);

    std::ofstream os2(fm.addPath("imitator_dataset.log"));
    s.data.writeToStream(os2);

    std::ofstream os3(fm.addPath("fitted_dataset.log"));
    f.data.writeToStream(os3);

    std::ofstream os4(fm.addPath("basis.log"));
    for(int i = 0; i < basis.size(); i++)
    {
    	basis[i]->writeOnStream(os4);
    }

    // print basis function used
    cout << basis.size() << std::endl;

    return 0;
}
