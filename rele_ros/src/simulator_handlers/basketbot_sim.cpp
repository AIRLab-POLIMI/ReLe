// Copyright 2006-2014 Dr. Marc Andreas Freese. All rights reserved.
// marc@coppeliarobotics.com
// www.coppeliarobotics.com
//
// -------------------------------------------------------------------
// This file is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
// You are free to use/modify/distribute this file for whatever purpose!
// -------------------------------------------------------------------
//
// This file was automatically created for V-REP release V3.1.1 on March 26th 2014


#include <stdio.h>
#include <stdlib.h>
#include <ros/ros.h>
#include "v_repConst.h"
#include <geometry_msgs/Twist.h>
// Used data structures:
#include "vrep_common/ProximitySensorData.h"
#include "vrep_common/VrepInfo.h"
#include "vrep_common/JointSetStateData.h"

// Used API services:
#include "vrep_common/simRosGetObjectHandle.h"
#include "vrep_common/simRosEnablePublisher.h"
#include "vrep_common/simRosEnableSubscriber.h"

// Global variables (modified by topic subscribers):
bool simulationRunning=true;
bool sensorTrigger=false;
float simulationTime=0.0f;

int leftMotorHandle;
int rightMotorHandle;
ros::Publisher motorSpeedPub;

// Topic subscriber callbacks:
void infoCallback(const vrep_common::VrepInfo::ConstPtr& info)
{
    simulationTime=info->simulationTime.data;
    simulationRunning=(info->simulatorState.data&1)!=0;
    if(!simulationRunning)
        ros::shutdown();
}

void sensorCallback(const vrep_common::ProximitySensorData::ConstPtr& sens)
{
    // We don't care about the detected distance here, we just trigger!
    sensorTrigger=true;
}
void chatterCallback(const geometry_msgs::Twist::ConstPtr& msg)
{
    std::cout << "velocita lineare: " << msg->linear.x<<"  velocita angolare: " << msg->angular.z<<std::endl;
    vrep_common::JointSetStateData motorSpeeds;


    float x = 10.0*msg->linear.x;
    float r = msg->angular.z;

    float desiredLeftMotorSpeed = x-r;
    float desiredRightMotorSpeed = x +r;


    // publish the motor speeds:
    motorSpeeds.handles.data.push_back(leftMotorHandle);
    motorSpeeds.handles.data.push_back(rightMotorHandle);
    motorSpeeds.setModes.data.push_back(2); // 2 is the speed mode
    motorSpeeds.setModes.data.push_back(2);
    motorSpeeds.values.data.push_back(desiredLeftMotorSpeed);
    motorSpeeds.values.data.push_back(desiredRightMotorSpeed);
    motorSpeedPub.publish(motorSpeeds);

}

bool getHandle(std::string name,int& variable)
{
    ros::NodeHandle n;
    ros::ServiceClient client = n.serviceClient<vrep_common::simRosGetObjectHandle>("/vrep/simRosGetObjectHandle");
    vrep_common::simRosGetObjectHandle robot_handle;
    robot_handle.request.objectName = name;
    if(!client.call(robot_handle))
    {
        std::cout <<"error in service call"<<std::endl;
        return false;
    }
    if(robot_handle.response.handle < 0)
    {
        std::cout <<"error, unable to get the handle of the vrep object "<<name<<std::endl;
        return false;
    }
    variable = robot_handle.response.handle;

    std::cout << name<<" handle: " <<robot_handle.response.handle<<std::endl;

    return true;

}

// Main code:
int main(int argc,char* argv[])
{
    // The joint handles and proximity sensor handles are given in the argument list
    // (when V-REP launches this executable, V-REP will also provide the argument list)

    /*if (argc>=3) {
    	leftMotorHandle=atoi(argv[1]);
    	rightMotorHandle=atoi(argv[2]);
    	std::cout << "argomenti: "<<argv[1]<< " , " << argv[2]<<std::endl;
    } else {
    	printf("Indicate following arguments: 'leftMotorHandle rightMotorHandle sensorHandle'!\n");
    	sleep(5000);
    	return 0;
    }*/

    // Create a ROS node. The name has a random component:
    int _argc = 0;
    char** _argv = NULL;
    struct timeval tv;
    unsigned int timeVal=0;
    if (gettimeofday(&tv,NULL)==0)
        timeVal=(tv.tv_sec*1000+tv.tv_usec/1000)&0x00ffffff;
    std::string nodeName("basketbot_sim");
    std::string randId(boost::lexical_cast<std::string>(timeVal+int(999999.0f*(rand()/(float)RAND_MAX))));
    nodeName+=randId;
    ros::init(_argc,_argv,nodeName.c_str());

    getHandle("left_wheel_joint",leftMotorHandle);
    getHandle("right_wheel_joint",rightMotorHandle);
    //getHandle("base_link_respondable",robotHandle);

    if(!ros::master::check())
        return(0);

    ros::NodeHandle node("~");
    printf("rosBubbleRob just started with node name %s\n",nodeName.c_str());

    // 1. Let's subscribe to V-REP's info stream (that stream is the only one enabled by default,
    // and the only one that can run while no simulation is running):
    ros::Subscriber subInfo=node.subscribe("/vrep/info",1,infoCallback);


    // 4. Let's tell V-REP to subscribe to the motor speed topic (publisher to that topic will be created further down):
    ros::ServiceClient client_enableSubscriber=node.serviceClient<vrep_common::simRosEnableSubscriber>("/vrep/simRosEnableSubscriber");
    vrep_common::simRosEnableSubscriber srv_enableSubscriber;

    srv_enableSubscriber.request.topicName="/"+nodeName+"/wheels"; // the topic name
    srv_enableSubscriber.request.queueSize=1; // the subscriber queue size (on V-REP side)
    srv_enableSubscriber.request.streamCmd=simros_strmcmd_set_joint_state; // the subscriber type
    if ( client_enableSubscriber.call(srv_enableSubscriber)&&(srv_enableSubscriber.response.subscriberID!=-1) )
    {
        // ok, the service call was ok, and the subscriber was succesfully started on V-REP side
        // V-REP is now listening to the desired motor joint states

        // 5. Let's prepare a publisher of those motor speeds:
        motorSpeedPub=node.advertise<vrep_common::JointSetStateData>("wheels",1);

        // 6. Finally we have the control loop:
        float driveBackStartTime=-99.0f;


        ros::NodeHandle n;
        ros::Subscriber sub = n.subscribe("cmd_vel", 1000, chatterCallback);


        while (ros::ok()&&simulationRunning)
        {
            // this is the control loop (very simple, just as an example)

            // handle ROS messages:
            ros::spin();


        }
    }
    else
        std::cout <<"unable to subscribe"<<std::endl;

    ros::shutdown();
    printf("rosBubbleRob just ended!\n");
    return(0);
}
