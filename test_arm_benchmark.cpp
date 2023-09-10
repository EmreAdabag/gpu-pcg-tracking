#include <iostream>

#define EXAMPLE_ROBOT_DATA_MODEL_DIR "/home/a2rlab/anaconda3/envs/crocoddyl/share/example-robot-data/robots"

#include "crocoddyl/benchmark/factory/arm.hpp"

int main() {
    // Initialize shared_ptr objects for runningModel and terminalModel
    boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<float>> runningModel;
    boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<float>> terminalModel;

    // Call the build_arm_action_models function
    crocoddyl::benchmark::build_arm_action_models<float>(runningModel, terminalModel);

    // Check if the models have been properly built (if applicable)
    if (runningModel && terminalModel) {
        std::cout << "Done" << std::endl;
    }
    return 0;
}
