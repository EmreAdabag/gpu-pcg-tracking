/***
nvcc -o printLCM.exe printLCM.cu -llcm -lpthread -gencode arch=compute_89,code=sm_89 -O3
***/

#include <thread>
#include <pthread.h>
#include <lcm/lcm-cpp.hpp>
#include "lcmtypes/drake/lcmt_iiwa_status.hpp"
#include "lcmtypes/drake/lcmt_iiwa_command.hpp"
#include "lcmtypes/drake/lcmt_robot_plan.hpp"
const char* const kLcmStatusChannel = "IIWA_STATUS";
const char* const kLcmCommandChannel = "IIWA_COMMAND";
const char* const kLcmPlanChannel = "COMMITTED_ROBOT_PLAN";

template <typename T>
class LCM_IIWA_STATUS_printer {
    public:
        LCM_IIWA_STATUS_printer(){}
        ~LCM_IIWA_STATUS_printer(){}

        void handleMessage(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const drake::lcmt_iiwa_status *msg){
            printf("%ld | %f %f %f %f %f %f %f | %f %f %f %f %f %f %f \n",msg->utime,
                msg->joint_position_measured[0],msg->joint_position_measured[1],msg->joint_position_measured[2],msg->joint_position_measured[3],
                msg->joint_position_measured[4],msg->joint_position_measured[5],msg->joint_position_measured[6],
                msg->joint_velocity_estimated[0],msg->joint_velocity_estimated[1],msg->joint_velocity_estimated[2],msg->joint_velocity_estimated[3],
                msg->joint_velocity_estimated[4],msg->joint_velocity_estimated[5],msg->joint_velocity_estimated[6]
            );
        }
};

template <typename T>
__host__
void run_IIWA_STATUS_printer(LCM_IIWA_STATUS_printer<T> *handler){
    lcm::LCM lcm_ptr; if(!lcm_ptr.good()){printf("LCM Failed to init\n");}
    lcm::Subscription *sub = lcm_ptr.subscribe(kLcmStatusChannel, &LCM_IIWA_STATUS_printer<T>::handleMessage, handler);
    sub->setQueueCapacity(1);
    while(0 == lcm_ptr.handle());
}


int main(int argc, char *argv[])
{
    // init rand
    auto handler = new LCM_IIWA_STATUS_printer<float>();
    auto thread  = std::thread(&run_IIWA_STATUS_printer<float>, handler);
    printf("Thread Launched!\n");
    thread.join();
    return 0;
}