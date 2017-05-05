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

#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/Joy.h>

using namespace ReLe;

#define WAVELETS
#define MEYER

const double maxT = 10.0;

void publishSetpoint(ros::Publisher& p, arma::vec& v)
{
	geometry_msgs::Twist msg;

	msg.linear.x = v(0);
	msg.linear.y = v(1);
	msg.angular.z = v(2);

	p.publish(msg);
}

bool stop;
bool next;

void joyCallback(const sensor_msgs::Joy::ConstPtr& joy_msg)
{
	if(joy_msg->buttons[2])
	{
		stop = true;
		std::cout << "STOPPED!" << std::endl;
	}
	else if(joy_msg->buttons[1])
	{
		next = true;
		std::cout << "NEXT!" << std::endl;
	}
}

int main(int argc, char *argv[])
{

	ros::init(argc, argv, "emotion_runner");

	ros::NodeHandle nh;

	ros::Publisher p = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
	ros::Subscriber s = nh.subscribe<sensor_msgs::Joy>("joy", 1, joyCallback);
	stop = false;
	next = false;

	if(argc < 2)
	{
		std::cout << "Missing emotion name parameter" << std::endl;
		return -1;
	}

	std::string emotionName(argv[1]);

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
    std::cout << "Generating emotions trajectories" << std::endl;
    std::cout << "Emotion: " << emotionName << std::endl;

    FileManager fm("emotions", emotionName);

    //Load emotions parameters
    arma::mat theta;
    theta.load(fm.addPath("theta.txt"), arma::raw_ascii);

    std::cout << "Trajectories: " << theta.n_cols << std::endl;

    //Run emotion
    CollectorStrategy<DenseAction, DenseState> f;

    //Compute fitted trajectory for each demonstration
    for(int i = 0; i < theta.n_cols && ros::ok(); i++)
    {
    	policy.setParameters(theta.col(i));

    	arma::vec t(1, arma::fill::zeros);
    	const double dt = 1e-2;

    	ros::Rate r(1.0/dt);
    	for(int i = 0; i < 2000 && ros::ok() && !stop && !next; i++)
    	{
    		arma::vec v = policy(t);
    		publishSetpoint(p, v);

    		t(0) += dt;
    		r.sleep();
    		ros::spinOnce();
    	}

    	if(stop)
    	{
    		while(!next)
    		{
    			ros::spinOnce();
    			usleep(200);
    		}

    		stop = false;
    		next = false;
    	}

    	if(next)
    	{
    		next = false;
    	}
	}

	// Save the dataset in ReLe format
    std::ofstream os(fm.addPath("imitator_dataset.log"));
    f.data.writeToStream(os);


}
