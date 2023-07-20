/***
nvcc -std=c++11 -o runMPC_LCM.exe runMPC_LCM.cu -llcm -lpthread -gencode arch=compute_61,code=sm_61 -O3
***/
#include <thread>
#include <pthread.h>
#include <lcm/lcm-cpp.hpp>

const unsigned STEPS_IN_TRAJ = 8;
#define TORQUE_CONTROL true

#include "interface.cuh"

#include "lcmtypes/drake/lcmt_iiwa_status.hpp"
#include "lcmtypes/drake/lcmt_iiwa_command.hpp"
#include "lcmtypes/drake/lcmt_robot_plan.hpp"
const char* const kLcmStatusChannel = "IIWA_STATUS";
const char* const kLcmCommandChannel = "IIWA_COMMAND";
const char* const kLcmPlanChannel = "COMMITTED_ROBOT_PLAN";





//
//
// TODO: add in SQP call here and the right input / output functions
//
//
// template <typename T>
// __host__
// void SQP_PCG(MPC_variables<T> *vars){
//     printf("hi\n");
// 	return;
// }

#include <iostream>

// void print_lcmt_robot_plan(const drake::lcmt_robot_plan& robot_plan) {
//     std::cout << "Robot Plan Utime: " << robot_plan.utime << std::endl;
//     std::cout << "Number of States: " << robot_plan.num_states << std::endl;

//     for(int i = 0; i < robot_plan.plan.size(); i++) {
//         std::cout << "State " << i+1 << ":" << std::endl;
//         std::cout << "\tState Utime: " << robot_plan.plan[i].utime << std::endl;
//         std::cout << "\tNumber of Joints: " << robot_plan.plan[i].num_joints << std::endl;
        
//         for(int j = 0; j < robot_plan.plan[i].num_joints; j++) {
//             std::cout << "\tJoint " << j+1 << ":" << std::endl;
//             std::cout << "\t\tJoint Name: " << robot_plan.plan[i].joint_name[j] << std::endl;
//             std::cout << "\t\tJoint Position: " << robot_plan.plan[i].joint_position[j] << std::endl;
//         }
//     }
// }


template <typename T>
class LCM_MPCLoop_Handler {
    public:
        MPC_variables<T> *vars; // local pointer to the global algorithm variables
        lcm::LCM lcm_ptr; // ptr to LCM object for publish ability
        bool first_utime_is_set;

        // init and store the global variables
        LCM_MPCLoop_Handler(MPC_variables<T> *vars_in) : vars(vars_in){
            first_utime_is_set = false;
            if(!lcm_ptr.good()){
                printf("LCM Failed to Init in Traj Runner\n"); vars = nullptr;
            }
        }
        // do nothing in the destructor
        ~LCM_MPCLoop_Handler(){}
    
        // lcm callback function for new arm status
        void handleStatus(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const drake::lcmt_iiwa_status *msg){
            
            // if (first_utime_is_set){ return; }
            if (! first_utime_is_set ){
                first_utime_is_set = true;
                vars->first_utime = msg->utime;
            }


            // load in the new status
            vars->utime = msg->utime;
            // std::cout << "got utime input " << vars->utime << std::endl;
             for (int i=0; i<grid::NUM_JOINTS; i++){
                vars->h_xs[i]                   = (T)(msg->joint_position_measured[i]); 
                vars->h_xs[i+grid::NUM_JOINTS]  = (T)(msg->joint_velocity_estimated[i]);
            }
            // std::cout << "Printing at start: " << std::endl;
            // std::cout << vars->h_xs[0] << " " << vars->h_xs[1] << " " << vars->h_xs[2] << " " << vars->h_xs[3] << " " << vars->h_xs[4] << " " << vars->h_xs[5] << " " << vars->h_xs[6] << " " << std::endl;

            // run the solver
            updateTrajectory<T>(vars);

            // T testarr[7] = {-1.806143, -1.452489, 2.101637, -1.246232, 1.675261, -0.489622, -0.368079};
            std::string namearr[7] = {"iiwa_joint_1", "iiwa_joint_2", "iiwa_joint_3", "iiwa_joint_4", "iiwa_joint_5", "iiwa_joint_6", "iiwa_joint_7"};

            // set up output package
            drake::lcmt_robot_plan dataOut;
            dataOut.utime = 0;
            // resize the output package
            dataOut.num_states = STEPS_IN_TRAJ;
            dataOut.plan.resize(dataOut.num_states);
            // copy over the values
            unsigned offset = TORQUE_CONTROL ? 2*grid::NUM_JOINTS : 0;
            for (int i=0; i<STEPS_IN_TRAJ; i++){
                // std::cout << "length of joint position was " << dataOut.plan[i].joint_position.size() << std::endl;
                dataOut.plan[i].joint_position.resize(grid::NUM_JOINTS);
                dataOut.plan[i].joint_name.resize(grid::NUM_JOINTS);
                dataOut.plan[i].num_joints = grid::NUM_JOINTS;
                // std::cout << "length of joint position is now " << dataOut.plan[i].joint_position.size() << std::endl;
                //memcpy(dataOut.plan[i].joint_position.data(),&(vars->h_xu[i*3*grid::NUM_JOINTS + offset]),grid::NUM_JOINTS*sizeof(T));
                dataOut.plan[i].utime = 0 + i*vars->timestep_long;
                dataOut.plan[i].joint_position.assign(&(vars->h_xu[i*3*grid::NUM_JOINTS + offset]), &(vars->h_xu[i*3*grid::NUM_JOINTS + offset]) + grid::NUM_JOINTS);
                // dataOut.plan[i].joint_position.assign(&(testarr[0]), &(testarr[7]));
                dataOut.plan[i].joint_name.assign(&(namearr[0]), &(namearr[7]));

            }
            
            // std::cout << "Printing at end: " << std::endl;
            std::cout << vars->h_xu[0] << " " << vars->h_xu[1] << " " << vars->h_xu[2] << " " << vars->h_xu[3] << " " << vars->h_xu[4] << " " << vars->h_xu[5] << " " << vars->h_xu[6] << " " << std::endl;
            // std::cout << "got utime at end " << dataOut.utime << std::endl;

            // std::cout << "Printing the plan " << std::endl;

            // print_lcmt_robot_plan(dataOut);
            // publish to trajRunner
            lcm_ptr.publish(kLcmPlanChannel,&dataOut);
        }      
};

template <typename T>
__host__
void runMPCHandler(LCM_MPCLoop_Handler<T> *handler){
    lcm::LCM lcm_ptr; if(!lcm_ptr.good()){printf("LCM Failed to init in MPC handler runner\n");}
    lcm::Subscription *sub = lcm_ptr.subscribe(kLcmStatusChannel, &LCM_MPCLoop_Handler<T>::handleStatus, handler);
    sub->setQueueCapacity(1);
    while(0 == lcm_ptr.handle());
    // while(1){lcm_ptr.handle();usleep(1000);}
}

template <typename T>
__host__
int runMPC_LCM(){
	// allocate variables and load inital variables
    MPC_variables<T> vars;
    setupTracking_pcg<T>(&vars);


    // run the solver once to get things started
	updateTrajectory<T>(&vars);


	// then create the handler and launch the MPC thread
	auto mpcHandler = new LCM_MPCLoop_Handler<T>(&vars);
 	auto mpcThread  = std::thread(&runMPCHandler<T>, mpcHandler);    
    printf("Launch the plan runner and then the simulator/hardware!\n");
    mpcThread.join();
    

    cleanupTracking_pcg<T>(&vars);

    return 0;
}

int main(int argc, char *argv[])
{
	// init rand
	srand(time(NULL));
	return runMPC_LCM<float>();
}