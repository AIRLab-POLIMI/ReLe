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

const double maxT = 10.0;

//#define TEST

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
        double dt = 1e-2;

        while(tf < maxT)
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


unsigned int countSamples(std::vector<arma::mat>& samples)
{
	unsigned int count = 0;
	for(auto& sample : samples)
	{
		count += sample.n_cols;
	}

	return count;
}


int main(int argc, char *argv[])
{
    std::cout << std::setprecision(OS_PRECISION);

    //Read emotion datatset
    auto* t1 = new RosTopicInterface_<geometry_msgs::Twist>("/cmd_vel", true, true);
    std::vector<RosTopicInterface*> topics;
    topics.push_back(t1);

    std::string basePath = "/home/dave/Dropbox/Dottorato/Major/test/";


    //Count emotions
    unsigned int emotionCount = std::count_if(
            directory_iterator(basePath),
            directory_iterator(),
            static_cast<bool(*)(const path&)>(is_directory) );

    cout << "processing " << emotionCount << " emotions, one should be negative examples" << std::endl;


    //Create appropriate features
    double df = 0.2;
    double fE = 20.0;

    BasisFunctions basis = FrequencyBasis::generate(0, df, fE, df, true);
    BasisFunctions tmp = FrequencyBasis::generate(0, 0, fE, df, false);
    basis.insert(basis.end(), tmp.begin(), tmp.end());

    DenseFeatures phi(basis);


    //Load dataset and compute features
    std::vector<string> emotionNames;
    std::vector<arma::mat> inputTmp;
    arma::mat negativeTmp;

    boost::filesystem::directory_iterator end_itr;
    for(boost::filesystem::directory_iterator i(basePath); i != end_itr; ++i )
    {
        int count = 0;

        if(boost::filesystem::is_directory(i->status()))
        {
            RosDataset rosDataset(topics);

            std::string emotionName = i->path().filename().string();

#ifdef TEST
            if(emotionName != "negative_examples")
                continue;
#endif

            std::cout << "-----------------------------------------------------" << std::endl;
            std::cout << "Emotion: " << emotionName << std::endl;

            FileManager fm("emotions_classification", emotionName);
            fm.createDir();


            for(boost::filesystem::directory_iterator j(i->path()); j != end_itr; ++j )
            {
                if(boost::filesystem::is_regular_file(j->status()) &&
                        j->path().extension() == ".bag")
                {
                    cout << j->path().string() << endl;
                    rosDataset.readEpisode(j->path().string());
                }
            }

            preprocessDataset(rosDataset.getData());

            MLEDistributionLinear estimator(phi);
            estimator.compute(rosDataset.getData());

            auto theta = estimator.getParameters();

            if(emotionName == "negative_examples")
            	negativeTmp = theta;
            else
            {
            	inputTmp.push_back(theta);
            	emotionNames.push_back(emotionName);
            }

            // print basis function used
            cout << "Feature extracted!" << std::endl;
            std::cout << "-----------------------------------------------------" << std::endl;
        }

    }

    inputTmp.push_back(negativeTmp);

    //Create dataset
    int datasetSize = countSamples(inputTmp);
    int featuresSize = phi.rows()*3;

    arma::mat input(featuresSize, datasetSize);
    arma::mat output(emotionCount - 1, datasetSize, arma::fill::zeros);

    unsigned int start = 0;

    std::ofstream fs(basePath + "emotionLabels.txt");

    for(unsigned int i = 0; i < inputTmp.size(); i++)
    {
    	unsigned int delta = inputTmp[i].n_cols;
    	input.cols(start, start + delta -1) = inputTmp[i];
    	if(i+1 != inputTmp.size())
    	{
    		output.cols(start, start + delta -1).row(i) = arma::ones(1, delta);
    		fs << emotionNames[i] << " " << i << endl;
    	}
    	start += delta;
    }

    fs.close();

    auto bfs = IdentityBasis::generate(input.n_rows);
    DenseFeatures identity(bfs);
    EmptyTreeNode<arma::vec> emptyNode(arma::zeros(output.n_rows));

    std::vector<Function*> layerFunction;
    layerFunction.push_back(new ReLU());
    layerFunction.push_back(new ReLU());
    layerFunction.push_back(new Sigmoid());
    std::vector<unsigned int> layerNeurons = {100, 100, emotionCount - 1};
    FFNeuralNetwork regressor(identity, layerNeurons, layerFunction);

    regressor.getHyperParameters().Omega = new L2_Regularization();
    regressor.getHyperParameters().lambda = 1e-4;
    regressor.getHyperParameters().optimizator = new ScaledConjugateGradient<arma::vec>(50000);


    BatchDataSimple trainingData(input, output);
    regressor.trainFeatures(trainingData);

    double J = 0;

    //Evaluate
    for(unsigned int i = 0; i < input.n_cols; i++)
    {
    	J += std::pow(arma::norm(regressor(input.col(i)) - output.col(i)), 2);
    }

    J /= input.n_cols;

    std::cout << "J: " << J << std::endl;

    regressor.getParameters().save(basePath + "ClassifierWeights.txt", arma::raw_ascii);

    return 0;
}

