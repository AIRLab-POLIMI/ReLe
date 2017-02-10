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
#include <rele/approximators/basis/IdentityBasis.h>

#include <rele/policy/parametric/differentiable/NormalPolicy.h>
#include <rele/IRL/algorithms/MLEDistributionLinear.h>
#include <rele/utils/ArmadilloExtensions.h>
#include <rele/approximators/regressors/nn/Autoencoder.h>


#include <rele/utils/FileManager.h>

#include <rele/environments/EmptyEnv.h>
#include <rele/core/PolicyEvalAgent.h>
#include <rele/core/Core.h>

#include "rele_ros/bag/RosDataset.h"
#include "rele_ros/bag/message/RosGeometryInterface.h"

#include <boost/filesystem.hpp>

#include "CompressedPolicy.h"

using namespace std;
using namespace arma;
using namespace boost::filesystem;
using namespace ReLe;
using namespace ReLe_ROS;


const double maxT = 10.0;


#define WAVELETS
#define REDUCTION

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


arma::uvec findFeatures(const arma::mat& theta)
{
    arma::vec min = arma::min(theta, 1);
    arma::vec max = arma::max(theta, 1);

    return arma::find((max - min) > 0.1);
}

int main(int argc, char *argv[])
{
    std::cout << std::setprecision(OS_PRECISION);

    //Read emotion datatset
    auto* t1 = new RosTopicInterface_<geometry_msgs::Twist>("/cmd_vel", true, true);
    std::vector<RosTopicInterface*> topics;
    topics.push_back(t1);

    std::string basePath = "/home/dave/Dropbox/Dottorato/Major/test/";

    boost::filesystem::directory_iterator end_itr;
    for(boost::filesystem::directory_iterator i(basePath); i != end_itr; ++i )
    {
        RosDataset rosDataset(topics);

        int count = 0;

        if(boost::filesystem::is_directory(i->status()))
        {
            std::string emotionName = i->path().filename().string();

#ifdef TEST
            if(emotionName!= "arrabbiato")
                continue;
#endif
            std::cout << "-----------------------------------------------------" << std::endl;
            std::cout << "Emotion: " << emotionName << std::endl;

            FileManager fm("emotions", emotionName);
            fm.createDir();


            for(boost::filesystem::directory_iterator j(i->path()); j != end_itr; ++j )
            {
                if(boost::filesystem::is_regular_file(j->status()) &&
                        j->path().extension() == ".bag")
                {
                    //cout << count++ << std::endl;
                    cout << j->path().string() << endl;
                    rosDataset.readEpisode(j->path().string());
                }
            }


            preprocessDataset(rosDataset.getData());

            //Create basis function for policy
            int uDim = 3;
#ifdef WAVELETS
            BasisFunctions basis = HaarWavelets::generate(0, 5, maxT);
#else
            unsigned int N = rosDataset.getData().getTransitionsNumber();
            double df = 0.1;
            double fE = 20.0;

            std::cout << "df: " << df << " fe: " << fE << " N: " << N << " tmax: "
                      << maxT << " 1/tmax: " << 1.0/maxT << endl;

            BasisFunctions basis = FrequencyBasis::generate(0, df, fE, df, true);
            BasisFunctions tmp = FrequencyBasis::generate(0, 0, fE, df, false);
            basis.insert(basis.end(), tmp.begin(), tmp.end());
#endif

            SparseFeatures phi(basis, uDim);

            //Fit Normal distribution
#ifdef WAVELETS
            BasisFunctions basisEst = HaarWavelets::generate(0, 5, maxT);
#else
            BasisFunctions basisEst = FrequencyBasis::generate(0, df, fE, df, true);
            BasisFunctions tmpEst = FrequencyBasis::generate(0, 0, fE, df, false);
            basisEst.insert(basisEst.end(), tmpEst.begin(), tmpEst.end());
#endif

            DenseFeatures phiEst(basisEst);

            MLEDistributionLinear estimator(phiEst);
            estimator.compute(rosDataset.getData());

            auto theta = estimator.getParameters();


#ifdef REDUCTION
            arma::uvec indices = findFeatures(theta);
            arma::mat thetaRed = theta.rows(indices);

            double reductionFactor = 0.4;

            unsigned int reducedDim = thetaRed.n_rows*reductionFactor;

            std::cout << reducedDim << std::endl;


            auto basisEnc = IdentityBasis::generate(thetaRed.n_rows);
            DenseFeatures phiEnc(basisEnc);
            Autoencoder autoencoder(phiEnc, reducedDim);

            autoencoder.getHyperParameters().optimizator = new ScaledConjugateGradient<arma::vec>(10000);

            autoencoder.getHyperParameters().Omega = new L2_Regularization();
            autoencoder.getHyperParameters().lambda = 0.01;

            std::cout << "J0: " << autoencoder.computeJFeatures(thetaRed) << std::endl;

            autoencoder.trainFeatures(thetaRed);

            std::cout << "Jf: " << autoencoder.computeJFeatures(thetaRed) << std::endl;

            arma::mat thetaNew(reducedDim, theta.n_cols);

            for(unsigned int i = 0; i < theta.n_cols; i++)
            {
                thetaNew.col(i) = autoencoder.encode(thetaRed.col(i));
            }

            thetaNew.save(fm.addPath("thetaNew.txt"), arma::raw_ascii);
            thetaRed.save(fm.addPath("thetaRed.txt"), arma::raw_ascii);

            arma::mat Cov = arma::cov(thetaNew.t());
            arma::vec mean = arma::mean(thetaNew, 1);

            std::vector<Range> ranges;

            ranges.push_back(Range(-1.20, 1.2));
            ranges.push_back(Range(-1.20, 1.2));
            ranges.push_back(Range(-6.28, 6.28));

            CompressedPolicy policy(phi, indices, autoencoder, ranges);
#else
            arma::mat Cov = arma::cov(theta.t());
            arma::vec mean = arma::mean(theta, 1);

            MVNPolicy policy(phi, arma::eye(uDim, uDim)*1e-3);
#endif

            auto M = nearestSPD(Cov);
            ParametricNormal dist(mean, M);

            // Testing
            EmptyEnv env(uDim, 100.0);

            CollectorStrategy<DenseAction, DenseState> f;

            //Compute fitted trajectory for each demonstration
            for(int i = 0; i < theta.n_cols; i++)
            {
#ifdef REDUCTION
                policy.setParameters(thetaNew.col(i));
#else
                policy.setParameters(theta.col(i));
#endif
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


            // Save the dataset in ReLe format
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

        }

    }

    return 0;
}
