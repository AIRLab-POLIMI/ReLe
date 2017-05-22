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

#include <rele/approximators/features/SparseFeatures.h>
#include <rele/approximators/basis/HaarWavelets.h>
#include <rele/approximators/basis/MeyerWavelets.h>

#include <rele/policy/parametric/differentiable/NormalPolicy.h>

#include <rele/environments/EmptyEnv.h>
#include <rele/core/Core.h>
#include <rele/core/PolicyEvalAgent.h>

#include <rele/utils/FileManager.h>

#include <boost/filesystem.hpp>

using namespace ReLe;

#define WAVELETS
#define PCA
//#define MEYER

const double maxT = 10.0;

int main(int argc, char *argv[])
{
    //Create basis function for policy
    int uDim = 3;
#ifdef WAVELETS
#ifdef MEYER
            MeyerWavelets wavelet;
#else
            HaarWavelets wavelet;
#endif
    BasisFunctions basis = Wavelets::generate(wavelet, 0, 5, maxT);
#else
    double df = 0.1;
    double fE = 20.0;

    std::cout << "df: " << df << " fe: " << fE << " N: " << " tmax: "
              << maxT << " 1/tmax: " << 1.0/maxT << std::endl;

    BasisFunctions basis = FrequencyBasis::generate(0, df, fE, df, true);
    BasisFunctions tmp = FrequencyBasis::generate(0, 0, fE, df, false);
    basis.insert(basis.end(), tmp.begin(), tmp.end());
#endif

    SparseFeatures phi(basis, uDim);

    MVNPolicy policy(phi, arma::eye(uDim, uDim)*1e-5);
    EmptyEnv env(uDim, 100.0);


    std::string basePath = "/tmp/ReLe/emotions/";

    std::cout << "==========================================================" << std::endl;
    std::cout << "Generating emotions from parameters" << std::endl;

    boost::filesystem::directory_iterator end_itr;
    for(boost::filesystem::directory_iterator i(basePath); i != end_itr; ++i )
    {

        if(boost::filesystem::is_directory(i->status()))
        {
            //Get emotion path and name
            std::string emotionName = i->path().filename().string();

            if(emotionName == "negative_examples" ||
                    emotionName == "training" ||
                    emotionName == "model")
                continue;

            std::cout << "-----------------------------------------------------" << std::endl;
            std::cout << "Emotion: " << emotionName << std::endl;

            FileManager fm("emotions", emotionName);

            //Load emotions parameters
            arma::mat theta;
            theta.load(fm.addPath("theta.txt"), arma::raw_ascii);

            std::cout << "Trajectories: " << theta.n_cols << std::endl;

            //Run emotion
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

            // Save the dataset in ReLe format
            std::ofstream os(fm.addPath("imitator_dataset.log"));
            f.data.writeToStream(os);

#ifdef PCA
            //Load emotions parameters
            arma::mat theta_pca;
            theta_pca.load(fm.addPath("theta_pca.txt"), arma::raw_ascii);

            std::cout << "Trajectories: " << theta_pca.n_cols << std::endl;

            //Run emotion
            CollectorStrategy<DenseAction, DenseState> f2;

            //Compute fitted trajectory for each demonstration
            for(int i = 0; i < theta_pca.n_cols; i++)
            {
                policy.setParameters(theta_pca.col(i));
                PolicyEvalAgent<DenseAction, DenseState> agent(policy);
                Core<DenseAction, DenseState> core(env, agent);

                core.getSettings().episodeLength = 2000;
                core.getSettings().loggerStrategy = &f2;
                core.runTestEpisode();
            }

            // Save the dataset in ReLe format
            std::ofstream os2(fm.addPath("pca_dataset.log"));
            f2.data.writeToStream(os2);

#endif

        }
    }
}
