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

#include "Segway.h"

using namespace std;
using namespace boost::numeric::odeint;

namespace ReLe
{

SegwaySettings::SegwaySettings()
{
    SegwaySettings::defaultSettings(*this);
}

void SegwaySettings::defaultSettings(SegwaySettings& settings)
{
    //Environment Parameters
    settings.gamma = 0.99;
    settings.continuosStateDim = 3;
    settings.continuosActionDim = 1;
    settings.rewardDim = 1;
    settings.finiteStateDim = 0;
    settings.finiteActionDim = 0;
    settings.isFiniteHorizon = false;
    settings.isAverageReward = false;
    settings.isEpisodic = true;
    settings.horizon = 300;

    //UWV Parameters
    settings.Mp = 10;
    settings.Mr = 15;
    settings.Ip = 19; //TODO change
    settings.Ir = 19; //TODO change
    settings.l = 1.2; //m
    settings.r = 0.2; //TODO change
    settings.dt = 0.03; //s
}

SegwaySettings::~SegwaySettings()
{

}

void SegwaySettings::WriteToStream(std::ostream& out) const
{
    //TODO implement
}

void SegwaySettings::ReadFromStream(std::istream& in)
{
    //TODO implement
}

Segway::SegwayOde::SegwayOde(SegwaySettings& config) :
    l(config.l), r(config.r), Ir(config.Ir), Ip(config.Ip), Mp(config.Mp), Mr(config.Mr), action(0)
{

}

void Segway::SegwayOde::operator ()(const state_type& x, state_type& dx,
                                    const double /* t */)
{
    //Status and actions
    const double tau = action;
    const double theta = x[0];
    const double omegaP = x[1];
    const double omegaR = x[2];

    //parameters
    const double h1 = (Mr+Mp)*pow(r, 2)+Ir;
    const double h2 = Mp*r*cos(theta);
    const double h3 = pow(l, 2)*Mp+Ip;

    //dinamics
    //angular velocity [rad/s]
    const double dTheta = omegaP;

    //angular acceleration [rad/s^2]
    const double dOmegaP = (h3 * l * Mp * r * sin(theta)
                            * pow(omegaP, 2) - g * h1 * l * Mp * sin(theta)
                            + (h3 + h1) * tau) / (pow(h3, 2) - h1 * h2);

    //angular acceleration [rad/s^2]
    const double dOmegaR = (h2 * l * Mp * r * sin(theta)
                            * pow(omegaP, 2) - g * h3 * l * Mp * sin(theta)
                            + (h3 + h2) * tau) / (pow(h3, 2) - h1 * h2);

    dx[0] = dTheta;
    dx[1] = dOmegaP;
    dx[2] = dOmegaR;
}

///////////////////////////////////////////////////////////////////////////////////////
/// SEGWAY ENVIRONMENT
///////////////////////////////////////////////////////////////////////////////////////

Segway::Segway()
    : ContinuousMDP(new SegwaySettings()), cleanConfig(true), segwayode(static_cast<SegwaySettings&>(getWritableSettings())),
      controlled_stepper (make_controlled< error_stepper_type >( 1.0e-6 , 1.0e-6 ))
{
    segwayConfig = static_cast<SegwaySettings*>(settings);
    currentState.set_size(segwayConfig->continuosStateDim);
}

Segway::Segway(SegwaySettings& config)
    : ContinuousMDP(&config), cleanConfig(false), segwayConfig(&config), segwayode(*segwayConfig),
      controlled_stepper (make_controlled< error_stepper_type >( 1.0e-6 , 1.0e-6 ))
{
    currentState.set_size(this->getSettings().continuosStateDim);
}

struct push_back_state_and_time
{
    std::vector< arma::vec >& m_states;
    std::vector< double >& m_times;

    push_back_state_and_time( std::vector< arma::vec > &states , std::vector< double > &times )
        : m_states( states ) , m_times( times ) { }

    void operator()( const arma::vec& x , double t )
    {
        m_states.push_back( x );
        m_times.push_back( t );
    }
};

void Segway::step(const DenseAction& action, DenseState& nextState, Reward& reward)
{
    double u = action[0];

    vector<state_type> x_vec;
    vector<double> times;

    //ODEINT (BOOST 1.53+)
    segwayode.action = u;
    double t0 = 0;
    double t1 = segwayConfig->dt;
//    integrate_adaptive(controlled_stepper , segwayode , currentState, t0 , t1 , t1/1000.0,
//                       push_back_state_and_time( x_vec , times ));
    integrate(segwayode , currentState, t0 , t1 , t1/1000.0,
              push_back_state_and_time( x_vec , times ));

    /* output */
    for( size_t i=0; i< x_vec.size(); i++ )
    {
        cerr << times[i] << '\t' << x_vec[i][0] << '\n';
    }

    nextState = currentState;

    //compute reward
    if(abs(currentState[0]) > M_PI/18)
    {
        currentState.setAbsorbing();
        reward[0] = -1000;
    }
    else
    {
        arma::mat Q(3, 3, arma::fill::eye);
        arma::mat R(1, 1, arma::fill::eye);


        const arma::vec& x = currentState;
        const arma::vec& u = action;
        arma::mat J = x.t()*Q*x + u.t()*R*u;

        reward[0] = -J[0];
    }

}

void Segway::getInitialState(DenseState &state)
{
    currentState.setAbsorbing(false);
    currentState[0] = 0.08;
    currentState[1] = 0;
    currentState[2] = 0;
    state = currentState;
}



}
