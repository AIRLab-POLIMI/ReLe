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
#include <rele/approximators/basis/PolynomialFunction.h>

#include <rele/policy/parametric/differentiable/NormalPolicy.h>
//#include <rele/policy/utils/MLE.h>
#include <rele/policy/utils/LinearStatisticEstimation.h>

#include <rele/utils/FileManager.h>

#include <rele/environments/EmptyEnv.h>
#include <rele/core/PolicyEvalAgent.h>
#include <rele/core/Core.h>

#include "rele_ros/bag/RosDataset.h"
#include "rele_ros/bag/message/RosGeometryInterface.h"


using namespace std;
using namespace arma;
using namespace ReLe;
using namespace ReLe_ROS;

void preprocessDataset(Dataset<DenseAction, DenseState>& data)
{
	for(unsigned int ep = 0; ep < data.size(); ep++)
	{
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

		double t0 = episode[0].x(0);

		for(i = 0; i < data[ep].size(); i++)
		{
			auto& tr = episode[i];

			tr.x(0) -= t0;
		}


		for(i = episode.size() - 1; i > 0; i--)
		{
			auto& tr = episode[i];

			if(tr.u(0) != 0 || tr.u(1) != 0 || tr.u(2) != 0)
			{
				break;
			}
		}

		episode.erase(episode.begin() + i + 1, episode.end());

	}

	/*unsigned int size = data[0].size();

	double tf = data[0].back().x(0);


	cout <<  size << endl;

	for(unsigned int i = 0; i < size; i++)
	{
		auto tr = data[0][i];
		tr.x += tf;
		data[0].push_back(tr);
	}


	cout <<  data[0].size() << endl;*/
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
    //std::string file = "triskar_2017-01-10-15-47-34.bag";
    //std::string file = "triskar_2017-01-10-15-46-18.bag";
    std::string file = "triskar_2017-01-10-16-01-03.bag";

    rosDataset.readEpisode(basePath+file);

    /*std::ofstream os0(fm.addPath("expert_dataset_original.log"));
    rosDataset.getData().writeToStream(os0);*/

    preprocessDataset(rosDataset.getData());

    double maxT = rosDataset.getData()[0].back().x(0);

    unsigned int N = rosDataset.getData().getTransitionsNumber();
    double df = 1/maxT;
    double fE = 20.0;
    int uDim = 3;

    std::cout << "df: " << df << " fe: " << fE << " N: " << N << " tmax: "
    			<< maxT << " 1/tmax: " << 1.0/maxT << endl;

    //Create basis function for policy
    BasisFunctions basis = FrequencyBasis::generate(0, df, fE, df, true);
   	BasisFunctions tmp = FrequencyBasis::generate(0, 0, fE, df, false);
    basis.insert(basis.end(), tmp.begin(), tmp.end());

    SparseFeatures phi(basis, uDim);

    //Fit Normal policy
    LinearStatisticEstimation estimator;
    BasisFunctions basisEst = FrequencyBasis::generate(0, df, fE, df, true);
    BasisFunctions tmpEst = FrequencyBasis::generate(0, 0, fE, df, false);
    basisEst.insert(basisEst.end(), tmpEst.begin(), tmpEst.end());
    //auto* tmp2Est = new PolynomialFunction();
    //basisEst.push_back(tmp2Est);

    DenseFeatures phiEst(basisEst);

    estimator.computeMLE(phiEst, rosDataset.getData());

    cout << "Cov:" << endl << estimator.getCovariance() << endl;

    //Create policy (add small elements on diagonal to avoid uncontrolled axis)
    MVNPolicy policy(phi, estimator.getCovariance()+arma::eye(uDim, uDim)*1e-10);
    policy.setParameters(estimator.getMeanParameters());

    //Test the fitted policy
    EmptyEnv env(uDim, 100.0);
    PolicyEvalAgent<DenseAction, DenseState> agent(policy);
    Core<DenseAction, DenseState> core(env, agent);

    CollectorStrategy<DenseAction, DenseState> s;
    core.getSettings().episodeLength = 2000;
    core.getSettings().loggerStrategy = &s;
    core.runTestEpisode();

    //Save the dataset in ReLe format
    std::ofstream os1(fm.addPath("expert_dataset.log"));
    rosDataset.getData().writeToStream(os1);

    std::ofstream os2(fm.addPath("imitator_dataset.log"));
    s.data.writeToStream(os2);



    return 0;
}
