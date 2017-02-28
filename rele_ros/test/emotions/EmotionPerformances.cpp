/*
 * rele_ros,
 *
 *
 * Copyright (C) 2017 Davide Tateo
 * Versione 1.0
 *
 * This file is part of rele_ros.
 *
 * rele_ros is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * rele_ros is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with rele_ros.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <rele/approximators/features/DenseFeatures.h>

#include <rele/environments/EmptyEnv.h>
#include <rele/core/Core.h>
#include <rele/approximators/regressors/trees/ExtraTreeEnsemble.h>
#include <rele/approximators/regressors/trees/KDTree.h>
#include <rele/approximators/regressors/nn/FFNeuralNetwork.h>
#include <rele/approximators/basis/FrequencyBasis.h>
#include <rele/approximators/basis/IdentityBasis.h>
#include <rele/IRL/algorithms/MLEDistributionLinear.h>

#include <rele/utils/FileManager.h>

#include <boost/filesystem.hpp>

#include "rele_ros/bag/RosDataset.h"
#include "rele_ros/bag/message/RosGeometryInterface.h"

using namespace std;
using namespace arma;
using namespace boost::filesystem;
using namespace ReLe;
using namespace ReLe_ROS;

map<string, int> loadLabels(string basePath)
{
	map<string, int> map;
	std::fstream fs(basePath + "emotionLabels");

	while(fs)
	{
		string name;
		int label;
		fs >> name >> label;
		map[name] = label;
	}

	return map;
}

int main(int argc, char *argv[])
{
    std::cout << std::setprecision(OS_PRECISION);

    std::string basePath = "/home/dave/Dropbox/Dottorato/Major/test/";

    //Load labels
    map<string, int> labels = loadLabels(basePath);


    //Create appropriate features
    double df = 0.2;
    double fE = 20.0;

    BasisFunctions basis = FrequencyBasis::generate(0, df, fE, df, true);
    BasisFunctions tmp = FrequencyBasis::generate(0, 0, fE, df, false);
    basis.insert(basis.end(), tmp.begin(), tmp.end());

    DenseFeatures phi(basis);

    //Load NN regressor
    auto bfs = IdentityBasis::generate(phi.rows()*3);
    DenseFeatures identity(bfs);

    std::vector<Function*> layerFunction;
    layerFunction.push_back(new ReLU());
    layerFunction.push_back(new ReLU());
    layerFunction.push_back(new Sigmoid());
    std::vector<unsigned int> layerNeurons = {100, 100, 6};
    FFNeuralNetwork regressor(identity, layerNeurons, layerFunction);

    arma::vec p;
    p.load(basePath + "ClassifierWeights.txt");
    regressor.setParameters(p);

    double Jtot = 0;


    //Load dataset and compute features
    std::vector<arma::mat> inputTmp;
    int count = 0;

    boost::filesystem::directory_iterator end_itr;
    for(boost::filesystem::directory_iterator i(basePath); i != end_itr; ++i )
    {

        if(boost::filesystem::is_directory(i->status()))
        {
            std::string emotionName = i->path().filename().string();

            if(emotionName == "negative_examples")
            	continue;

            std::cout << "Emotion: " << emotionName << std::endl;

            FileManager fm("emotions", emotionName);
            fm.createDir();

            Dataset<DenseAction, DenseState> dataset;

            std::fstream fs(fm.addPath("imitator_dataset.log"));
            dataset.readFromStream(fs);

            MLEDistributionLinear estimator(phi);
            estimator.compute(dataset);

            auto theta = estimator.getParameters();

            // print basis function used
            cout << "Feature extracted!" << std::endl;
            std::cout << "-----------------------------------------------------" << std::endl;


            double Jemotion = 0;
            int errors = 0;
            int tot = dataset.getEpisodesNumber();
            for(unsigned int i = 0; i < tot; i++)
            {
            	arma::vec res = regressor(theta.col(i));
            	arma::vec target(res.n_rows, arma::fill::zeros);
            	target(labels[emotionName]) = 1;
            	arma::vec delta = (res-target);
            	Jemotion += arma::as_scalar(delta.t()*delta);
            	errors += labels[emotionName] != res.index_max();
            	/*std:: cout << "res: " << res.t() << std::endl;
            	std:: cout << "target: " << target.t() << std::endl;*/
            }

            Jemotion /= tot;

            cout << "errors: " << errors << "/" << tot << " " << (float) (tot-errors) / tot << "%" << std::endl;
            cout << "emotion performace: " << Jemotion << std::endl;
            std::cout << "=====================================================" << std::endl;

            Jtot += Jemotion;
            count++;
        }

    }

    Jtot /= count;
    cout << "system performaces: " << Jtot << std::endl;
    std::cout << "=====================================================" << std::endl;

    return 0;
}
