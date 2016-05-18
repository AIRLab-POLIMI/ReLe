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

#include "rele/core/Core.h"
#include "rele/core/BatchCore.h"
#include "rele/core/PolicyEvalAgent.h"

#include "rele/environments/MountainCar.h"

#include "rele/statistics/DifferentiableNormals.h"
#include "rele/policy/nonparametric/RandomPolicy.h"

#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/basis/GaussianRbf.h"
#include "rele/approximators/basis/ConditionBasedFunction.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/approximators/regressors/others/LinearApproximator.h"

#include "rele/algorithms/batch/td/LSPI.h"

#include "rele/IRL/ParametricRewardMDP.h"
#include "rele/IRL/algorithms/EGIRL.h"
#include "rele/IRL/algorithms/LinearMLEDistribution.h"

#include "rele/utils/FileManager.h"

#include <boost/timer/timer.hpp>

using namespace boost::timer;
using namespace std;
using namespace ReLe;
using namespace arma;

class mountain_car_manual_policy : public ParametricPolicy<FiniteAction, DenseState>
{
public:
    unsigned int operator()(const arma::vec& state) override
    {
        if (RandomGenerator::sampleEvent(0.4))
        {
            return RandomGenerator::sampleUniformInt(0,1);
        }
        else
        {
            double speed = state(MountainCar::StateLabel::velocity);
            if (speed <= eps)
                return 0;
            else
                return 2;
        }
    }

    double operator()(const arma::vec& state, const unsigned int& action) override
    {
        double speed = state(MountainCar::StateLabel::velocity);
        if (speed <= eps && action == 0)
            return 1;
        else if(speed > eps && action == 2)
            return 1;
        else
            return 0;
    }

    //Parametric policy interface
public:
    virtual arma::vec getParameters() const override
    {
        arma::vec w = {eps};

        return w;
    }

    virtual const unsigned int getParametersSize() const override
    {
        return 1;
    }

    virtual void setParameters(const arma::vec& w) override
    {
        eps = w(0);
        std::cout << eps << std::endl;
    }


    // Policy interface
public:
    string getPolicyName() override
    {
        return "mountain_car_manual_policy";
    }

    string printPolicy() override
    {
        return "";
    }

    mountain_car_manual_policy* clone() override
    {
        return new mountain_car_manual_policy();
    }

private:
    double eps;
};

int main(int argc, char *argv[])
{
    if(argc != 3)
    {
        cout << "Wrong argument number: n_episode, n_experiment must be provided" << endl;
        return -1;
    }

    string n_episodes(argv[1]);
    string n_experiment(argv[2]);

    string path = "nips/mc/" + n_episodes + "/" +n_experiment + "/";

    FileManager fm(path + "EGIRL");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    // === define expert's policy === //
    mountain_car_manual_policy expertPolicy;

    arma::vec muExpert = {0};
    arma::mat SigmaExpert = { 1e-3 };
    ParametricNormal expertDist(muExpert, SigmaExpert);


    // === get expert's trajectories === //
    ifstream is;
    is.open("/tmp/ReLe/" + path + "trajectories.dat");
    Dataset<FiniteAction, DenseState> dataExpert;
    dataExpert.readFromStream(is);


    // === Estimate trajectories === //
    arma::mat theta(1, dataExpert.size());
    for(unsigned int i = 0; i < dataExpert.size(); i++)
    {
    	double minV = -std::numeric_limits<double>::infinity();
    	double maxV = -minV;
    	for(auto& tr : dataExpert[i])
    	{

    		if(tr.u == 0)
    		{
    			minV = std::max(tr.x(1), minV);
    		}
    		else
    		{
    			maxV = std::min(tr.x(1), maxV);
    		}
    	}

		if(std::isinf(minV) || std::isinf(maxV))
		{
			theta.col(i) = 0;
		}
		else
		{
			theta.col(i) = 0.5*(maxV + minV);
		}
    }

    std::cout << "theta" << std::endl;
    std::cout << theta << std::endl;


    // === recover reward by IRL === //

    vec pos_linspace = linspace(-1.2,0.6,7);
    vec vel_linspace = linspace(-0.07,0.07,7);

    arma::mat yy_vel, xx_pos;
    meshgrid(vel_linspace, pos_linspace, yy_vel, xx_pos);

    arma::vec pos_mesh = vectorise(xx_pos);
    arma::vec vel_mesh = vectorise(yy_vel);
    arma::mat XX = arma::join_horiz(vel_mesh,pos_mesh);


    double sigma_position = 2*pow((0.6+1.2)/4.0,2);
    double sigma_speed    = 2*pow((0.07+0.07)/4.0,2);


    arma::vec widths = {sigma_speed, sigma_position};
    arma::mat WW = repmat(widths, 1, XX.n_rows);
    arma::mat XT = XX.t();

    BasisFunctions rewardBasis = GaussianRbf::generate(XT, WW);
    DenseFeatures phiReward(rewardBasis);
    LinearApproximator rewardF(phiReward);

    cout << "Rewards size: " << rewardF.getParametersSize() << endl;

    EGIRL<FiniteAction,DenseState> irlAlg(dataExpert, theta, expertDist, rewardF,
                                          0.9, IrlEpGrad::PGPE_BASELINE);

    irlAlg.run();

    vec rewWeights = rewardF.getParameters();
    cout << "Weights (EGIRL): " << rewWeights.t();

    return 0;
}
