/***
nvcc -std=c++11 -o runMPC_LCM.exe runMPC_LCM.cu -llcm -lpthread -gencode arch=compute_61,code=sm_61 -O3
***/
#include <thread>
#include <pthread.h>
#include <lcm/lcm-cpp.hpp>

const unsigned STEPS_IN_TRAJ = 16;
const bool TORQUE_CONTROL_SWITCH = false;

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

template <typename T, bool TORQUE_CONTROL = false>
class LCM_MPCLoop_Handler {
    public:
        MPC_variables<T> *vars; // local pointer to the global algorithm variables
        lcm::LCM lcm_ptr; // ptr to LCM object for publish ability

        // init and store the global variables
        LCM_MPCLoop_Handler(MPC_variables<T> *vars_in) : vars(vars_in){
            if(!lcm_ptr.good()){
                printf("LCM Failed to Init in Traj Runner\n"); vars = nullptr;
            }
        }
        // do nothing in the destructor
        ~LCM_MPCLoop_Handler(){} 
    
        // lcm callback function for new arm status
        void handleStatus(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const drake::lcmt_iiwa_status *msg){
            
            // load in the new status
            vars->utime = msg->utime;
            for (int i=0; i<grid::NUM_JOINTS; i++){
                vars->h_xs[i]                   = (T)(msg->joint_position_measured)[i]; 
                vars->h_xs[i+grid::NUM_JOINTS]  = (T)(msg->joint_velocity_estimated)[i];
            }

            // run the solver
            updateTrajectory<T>(vars);

            // set up output package
            drake::lcmt_robot_plan dataOut;
            dataOut.utime = vars->utime;
            // resize the output package
            dataOut.num_states = grid::NUM_JOINTS*STEPS_IN_TRAJ;
            dataOut.plan.resize(dataOut.num_states);
            // copy over the values
            unsigned offset = TORQUE_CONTROL ? 2*grid::NUM_JOINTS : 0;
            for (int i=0; i<STEPS_IN_TRAJ; i++){
                memcpy(&(dataOut.plan[i * grid::NUM_JOINTS]),&(vars->h_xu[i*3*grid::NUM_JOINTS + offset]),grid::NUM_JOINTS*sizeof(T));        
            }
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

template <typename T, bool TORQUE_CONTROL = false>
__host__
int runMPC_LCM(){
	// allocate variables and load inital variables
    MPC_variables<T> vars;
    setupTracking_pcg<T>(&vars);


    // run the solver once to get things started
	updateTrajectory<T>(&vars);


	// then create the handler and launch the MPC thread
	auto mpcHandler = new LCM_MPCLoop_Handler<T,TORQUE_CONTROL>(&vars);
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
	return runMPC_LCM<float,TORQUE_CONTROL_SWITCH>();
}