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

        //set episode lenght to maxT seconds
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

        if(boost::filesystem::is_directory(i->status()))
        {
            std::string emotionName = i->path().filename().string();

            std::cout << "-----------------------------------------------------" << std::endl;
            std::cout << "Emotion: " << emotionName << std::endl;

            FileManager fm("emotions_features", emotionName);
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
            theta.save(fm.addPath("theta.txt"), arma::raw_ascii);

        }

    }

    return 0;
}
