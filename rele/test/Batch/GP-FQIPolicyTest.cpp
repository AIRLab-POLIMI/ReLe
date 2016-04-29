/*
 * rele,
 *
 *
 * Copyright (C) 2016 Davide Tateo
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

#include "rele/core/Core.h"
#include "rele/environments/MountainCar.h"
#include "rele/core/PolicyEvalAgent.h"
#include "rele/policy/q_policy/e_Greedy.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/environments/SwingPendulum.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/regressors/others/GaussianProcess.h"
#include "rele/utils/FileManager.h"

#include <iostream>

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{
    std::string env = argv[1];

    DenseMDP* mdp;

    if(env == "mc")
        mdp = new MountainCar(MountainCar::Ernst);
    else if(env == "ip")
        mdp = new DiscreteActionSwingUp;

    unsigned int stateDim = mdp->getSettings().stateDimensionality;
    unsigned int nActions = mdp->getSettings().actionsNumber;

	FileManager fm(env, "testFqi");
	fm.createDir();
	fm.cleanDir();

    BasisFunctions bfs = IdentityBasis::generate(stateDim);
    DenseFeatures phi(bfs);

    std::vector<std::string> algs = {"fqi", "dfqi", "wfqi"};
    std::string alg;
    unsigned int nExperiments = 20;
    unsigned int nEpisodes = 100;
    arma::mat Js(nExperiments, algs.size(), arma::fill::zeros);

    for(unsigned int a = 0; a < algs.size(); a++)
    {
    	alg = algs[a];

    	for(unsigned int e = 0; e < nExperiments; e++)
    	{
        	std::string testFileName = env + "-" + alg + "_" + std::to_string(e) + "Data.log";

    		std::string loadPath = "/home/tesla/Desktop/NIPS2016-GP/mc/" +
    				std::to_string(nEpisodes) + "Episodes/" + alg + "/";

			arma::mat hParams;
			hParams.load(loadPath + "hParams_" + std::to_string(e) + ".mat", arma::raw_ascii);
			arma::vec lengthScale = hParams.col(0);
			arma::vec rawSignalSigma = hParams.col(1);
			double signalSigma = arma::as_scalar(rawSignalSigma(arma::find(rawSignalSigma != arma::datum::inf)));

			std::vector<std::vector<GaussianProcess*>> gps;

			if(alg == "fqi" || alg == "wfqi")
			{
				std::vector<GaussianProcess*> gpsA;
				gps.push_back(gpsA);

				arma::mat alpha;
				arma::mat activeSetMat;
				alpha.load(loadPath + "alphas_" + std::to_string(e) + ".mat", arma::raw_ascii);
				activeSetMat.load(loadPath + "activeSetVectors_" + std::to_string(e) + ".mat", arma::raw_ascii);

				arma::cube activeSet(activeSetMat.n_rows, activeSetMat.n_cols / nActions, nActions);
				for(unsigned int a = 0; a < nActions; a++)
					activeSet.slice(a) = activeSetMat.cols(arma::span(stateDim * a, stateDim * a + stateDim - 1));

				for(unsigned int i = 0; i < alpha.n_cols; i++)
				{
					arma::vec rawAlphaVec = alpha.col(i);
					arma::vec alphaVec = rawAlphaVec(arma::find(rawAlphaVec != arma::datum::inf));

					GaussianProcess* gp = new GaussianProcess(phi);
					gp->getHyperParameters().lengthScale = lengthScale;
					gp->getHyperParameters().signalSigma = signalSigma;

					gp->setAlpha(alphaVec);

					arma::mat rawActiveSetMat = activeSet.slice(i);
					arma::vec temp = rawActiveSetMat.col(0);
					arma::mat activeSetMat = rawActiveSetMat.rows(arma::find(temp != arma::datum::inf));

					gp->setFeatures(activeSetMat);

					gps[0].push_back(gp);
				}
			}
			else if(alg == "dfqi")
			{
				std::vector<GaussianProcess*> gpsA;
				std::vector<GaussianProcess*> gpsB;
				gps.push_back(gpsA);
				gps.push_back(gpsB);

				arma::mat alphaMat;
				alphaMat.load(loadPath + "alphas_" + std::to_string(e) + ".mat", arma::raw_ascii);
				arma::cube alpha(alphaMat.n_rows, alphaMat.n_cols / nActions, nActions);
				for(unsigned int a = 0; a < nActions; a++)
					alpha.slice(a) = alphaMat.cols(arma::span(stateDim * a, stateDim * a + stateDim - 1));

				std::vector<arma::mat> activeSetsMat;
				arma::mat tempMat1;
				tempMat1.load(loadPath + "activeSetVectors1_" + std::to_string(e) + ".mat", arma::raw_ascii);
				activeSetsMat.push_back(tempMat1);
				arma::mat tempMat2;
				tempMat2.load(loadPath + "activeSetVectors2_" + std::to_string(e) + ".mat", arma::raw_ascii);
				activeSetsMat.push_back(tempMat2);
				arma::cube tempActiveSet(activeSetsMat[0].n_rows, activeSetsMat[0].n_cols / nActions, nActions, arma::fill::zeros);
				std::vector<arma::cube> activeSets = {tempActiveSet, tempActiveSet};

				for(unsigned int i = 0; i < 2; i++)
					for(unsigned int a = 0; a < nActions; a++)
						activeSets[i].slice(a) = activeSetsMat[i].cols(arma::span(stateDim * a, stateDim * a + stateDim - 1));

				for(unsigned int i = 0; i < 2; i++)
				{
					arma::mat rawAlphaMat = alpha.slice(i);

					for(unsigned int j = 0; j < alpha.n_cols; j++)
					{
						arma::vec rawAlphaVec = rawAlphaMat.col(j);
						arma::vec alphaVec = rawAlphaVec(arma::find(rawAlphaVec != arma::datum::inf));

						GaussianProcess* gp = new GaussianProcess(phi);
						gp->getHyperParameters().lengthScale = lengthScale;
						gp->getHyperParameters().signalSigma = signalSigma;

						gp->setAlpha(alphaVec);

						arma::mat rawActiveSetMat = activeSets[i].slice(j);
						arma::vec rawActiveSetVec = rawActiveSetMat.col(0);
						arma::mat activeSetMat = rawActiveSetMat.rows(
													 arma::find(rawActiveSetVec != arma::datum::inf));

						gp->setFeatures(activeSetMat);

						gps[i].push_back(gp);
					}
				}
			}

			e_GreedyMultipleRegressors policy(gps);
			policy.setEpsilon(0);
			policy.setNactions(nActions);
			PolicyEvalAgent<FiniteAction, DenseState> agent(policy);

			if(env == "mc")
			{
				unsigned int counter = 1;
				for(int i = -8; i <= 8; i++)
					for(int j = -8; j <= 8; j++)
					{
						double initialPosition = 0.125 * i;
						double initialVelocity = 0.375 * j;
						MountainCar testMdp(MountainCar::Ernst, initialPosition, initialVelocity);
						auto&& core = buildCore(testMdp, agent);
						core.getSettings().episodeLength = 100;
						core.getSettings().loggerStrategy =
							new WriteStrategy<FiniteAction, DenseState>(fm.addPath(testFileName));

						core.runTestEpisode();

						std::cout << counter++ << "/289" << std::endl;
					}

		    	arma::mat testEpisodes;
		    	testEpisodes.load(fm.addPath(testFileName), arma::csv_ascii);

		    	arma::uvec positiveIdxs = arma::find(testEpisodes.col(2 + stateDim + 1) == 1);
		    	unsigned int nPositives = positiveIdxs.n_elem;
		    	unsigned int nNegatives = 289 - nPositives;

		    	Js(e, a) = (nPositives - nNegatives) / 289;
			}
			else if(env == "ip")
			{
				auto&& core = buildCore(*mdp, agent);
				core.getSettings().episodeLength = 100;
				core.getSettings().loggerStrategy =
					new WriteStrategy<FiniteAction, DenseState>(fm.addPath(testFileName));

				core.runTestEpisode();

		    	arma::mat testEpisodes;
		    	testEpisodes.load(fm.addPath(testFileName), arma::csv_ascii);

		    	// calcolo reward medio per esperimento
			}

			delete mdp;
    	}
	}

    std::string savePath = "/home/tesla/Desktop/";
    std::string saveFileName = "Js-" + env + ".txt";
    Js.save(saveFileName, arma::raw_ascii);
}
