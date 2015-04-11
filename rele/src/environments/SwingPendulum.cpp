#include "../../include/rele/environments/SwingPendulum.h"

#include "RandomGenerator.h"
#include <cassert>

using namespace std;

namespace ReLe
{

SwingUpSettings::SwingUpSettings()
{
    SwingUpSettings::defaultSettings(*this);
}

void SwingUpSettings::defaultSettings(SwingUpSettings& settings)
{
    //Environment Parameters
    settings.gamma = 0.999;
    settings.continuosStateDim = 2;
    settings.continuosActionDim = -1;
    settings.rewardDim = 1;
    settings.finiteStateDim = -1;
    settings.finiteActionDim = 3;
    settings.isFiniteHorizon = false;
    settings.isAverageReward = false;
    settings.isEpisodic = false;
    settings.horizon = 80;

    //SwingUp parameters
    settings.stepTime = 0.01;
    double uMax = 2.0/*Doya's paper 5.0*/;
    double maxVelocity = M_PI_4 / settings.stepTime;
    settings.actionRange = Range(-uMax, uMax);
    settings.thetaRange = Range(-M_PI, M_PI);
    settings.velocityRange = Range(-maxVelocity, maxVelocity);
    settings.mass = 1.0;
    settings.length = 1.0;
    settings.g = 9.8;
    settings.requiredUpTime = 10.0;
    settings.upRange = M_PI_4 /*seconds*/;
    settings.useOverRotated = false;
    settings.random_start = false;

    settings.actionList =
    {
        settings.actionRange.Lo(), 0.0,
        settings.actionRange.Hi()
    };
    assert(settings.finiteActionDim == settings.actionList.size());
}

SwingUpSettings::~SwingUpSettings()
{

}

void SwingUpSettings::WriteToStream(ostream &out) const
{
    EnvironmentSettings::WriteToStream(out);
    out << stepTime << std::endl;
    out << actionRange.Lo() << " " << actionRange.Hi() << std::endl;
    out << thetaRange.Lo() << " " << thetaRange.Hi() << std::endl;
    out << velocityRange.Lo() << " " << velocityRange.Hi() << std::endl;
    out << mass << " " << length << " " << g << std::endl;
    out << requiredUpTime << " " << upRange << " ";
    out << (useOverRotated ? 1 : 0) << (random_start ? 1 : 0) << std::endl;

    int dima = actionList.size();
    out << dima;
    for (int i = 0; i < dima; ++i)
        out << " " << actionList[i];
}

void SwingUpSettings::ReadFromStream(istream &in)
{
    double lo, hi;
    int boolval;
    EnvironmentSettings::ReadFromStream(in);
    in >> stepTime;
    in >> lo >> hi;
    actionRange = Range(lo, hi);
    in >> lo >> hi;
    thetaRange = Range(lo, hi);
    in >> lo >> hi;
    velocityRange = Range(lo, hi);
    in >> mass >> length >> g >> requiredUpTime >> upRange;
    in >> boolval;
    useOverRotated = boolval == 1 ? true : false;
    in >> boolval;
    random_start = boolval == 1 ? true : false;

    int dima;
    in >> dima;
    for (int i = 0; i < dima; ++i)
    {
        in >> lo;
        actionList.push_back(lo);
    }

    assert(finiteActionDim == actionList.size());
}

///////////////////////////////////////////////////////////////////////////////////////
/// NLS ENVIRONMENTS
///////////////////////////////////////////////////////////////////////////////////////

DiscreteActionSwingUp::DiscreteActionSwingUp() :
    config()
{
    setupEnvirorment(config.continuosStateDim, config.finiteActionDim,
                     config.rewardDim, config.isFiniteHorizon, config.isEpisodic,
                     config.horizon, config.gamma);
    currentState.set_size(config.continuosStateDim);

    //variable initialization
    previousTheta = cumulatedRotation = overRotatedTime = 0;
    overRotated = false;
    upTime = 0;
}

DiscreteActionSwingUp::DiscreteActionSwingUp(SwingUpSettings& config) :
    config(config)
{
    setupEnvirorment(config.continuosStateDim, config.finiteActionDim,
                     config.rewardDim, config.isFiniteHorizon, config.isEpisodic,
                     config.horizon, config.gamma);
    currentState.set_size(config.continuosStateDim);

    //variable initialization
    previousTheta = cumulatedRotation = overRotatedTime = 0;
    overRotated = false;
    upTime = 0;
}

void DiscreteActionSwingUp::step(const FiniteAction& action,
                                 DenseState& nextState, Reward& reward)
{
    //get current state
    double theta = currentState[0];
    double velocity = currentState[1];

    //std::cout << a.at() << std::endl;
    double torque = config.actionList[action.getActionN()];
    double thetaAcc = -config.stepTime * velocity
                      + config.mass * config.g * config.length * sin(theta) + torque;
    velocity = config.velocityRange.bound(velocity + thetaAcc);
    theta += velocity * config.stepTime;
    adjustTheta(theta);
    upTime = fabs(theta) > config.upRange ? 0 : upTime + 1;

    //update current state
    currentState[0] = theta;
    currentState[1] = velocity;

    double signAngleDifference = std::atan2(std::sin(theta - previousTheta),
                                            std::cos(theta - previousTheta));
    cumulatedRotation += signAngleDifference;
    if (!overRotated && std::abs(cumulatedRotation) > 5.0f * M_PI)
        overRotated = true;
    if (overRotated)
        overRotatedTime += 1;
    previousTheta = theta;

    //###################### TERMINAL CONDITION REACHED ######################
    bool endepisode = false;
    if (config.useOverRotated)
        // Reinforcement Learning in Continuous Time and Space (Kenji Doya)
        endepisode =
            (overRotated && (overRotatedTime > 1.0 / config.stepTime)) ?
            true : false;
    //return upTime + 1 >= requiredUpTime / stepTime; // 1000 steps

    currentState.setAbsorbing(endepisode);
    nextState = currentState;

    //###################### REWARD ######################
    if (config.useOverRotated)
        // Reinforcement Learning in Continuous Time and Space (Kenji Doya)
        reward[0] = (!overRotated) ? cos(nextState[0]) : -1.0;
    else
        reward[0] = cos(nextState[0]);

}

void DiscreteActionSwingUp::getInitialState(DenseState& state)
{
    double theta;
    upTime = 0;
    if (config.random_start)
        theta = RandomGenerator::sampleUniform(config.thetaRange.Lo(),
                                               config.thetaRange.Hi());
    else
        theta = M_PI_2;
    adjustTheta(theta);
    previousTheta = theta;
    cumulatedRotation = theta;
    overRotated = false;
    overRotatedTime = 0;

    currentState[0] = theta;
    currentState[1] = 0.0;
    currentState.setAbsorbing(false);

    state = currentState;
}

}
