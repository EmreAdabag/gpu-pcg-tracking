#include "pinocchio/parsers/urdf.hpp"

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include <Eigen/Dense>
#include <pinocchio/multibody/model.hpp>
#include <crocoddyl/core/actuation/squashing/smooth-sat.hpp>
#include <crocoddyl/core/costs/residual.hpp>
#include <crocoddyl/core/integrator/euler.hpp>
#include <crocoddyl/core/solvers/ddp.hpp>
#include <crocoddyl/core/utils/callbacks.hpp>
#include <crocoddyl/core/costs/cost-sum.hpp>

#include <crocoddyl/core/activations/weighted-quadratic.hpp>

#include <crocoddyl/multibody/states/multibody.hpp>
#include <crocoddyl/multibody/actuations/full.hpp>
#include <crocoddyl/multibody/residuals/state.hpp>
#include <crocoddyl/core/residuals/control.hpp>
#include <crocoddyl/multibody/actions/free-fwddyn.hpp>

#include <iostream>

#include <experiment_helpers.cuh>
#include <settings.cuh>

// PINOCCHIO_MODEL_DIR is defined by the CMake but you can define your own directory here.
#ifndef PINOCCHIO_MODEL_DIR
#define PINOCCHIO_MODEL_DIR "path_to_the_model_dir"
#endif

#define KNOT_POINTS 32
#define DT .001
#define NUM_CONTROLS 7
#define NUM_STATES 14
#define EE_SIZE 6

#define QD_COST 0.1
#define Q_COST 0.1
#define R_COST 0.0001

int main(int argc, char **argv)
{
	using namespace pinocchio;

	// You should change here to set up your own URDF file or just pass it as an argument of this example.
	const std::string urdf_filename = (argc <= 1) ? PINOCCHIO_MODEL_DIR + std::string("/example-robot-data/robots/ur_description/urdf/ur5_robot.urdf") : argv[1];
	// Load the urdf model

	// NOTE: parsing the URDF only works with doubles, if we need to use floats we need to parse first and then cast
	// https://github.com/stack-of-tasks/pinocchio/issues/2037
	ModelTpl<double> model_double;
	pinocchio::urdf::buildModel(urdf_filename, model_double);
	pinocchio::ModelTpl<float> model = model_double.cast<float>();
	// load the model stored in model_double
	// ModelTpl<pcg_t> robot_model = model_double.cast<float>();

	// // // Create data required by the algorithms
	// DataTpl<pcg_t> data(robot_model);

	// // Create the state and actuation models
	boost::shared_ptr<crocoddyl::StateMultibodyTpl<pcg_t>> state =
		boost::make_shared<crocoddyl::StateMultibodyTpl<pcg_t>>(
			boost::make_shared<pinocchio::ModelTpl<pcg_t>>(model));
	// boost::shared_ptr<pinocchio::ModelTpl<float>> robot_model_ptr = boost::make_shared<pinocchio::ModelTpl<float>>(robot_model);
	// boost::shared_ptr<crocoddyl::StateMultibodyTpl<float>> state = boost::make_shared<crocoddyl::StateMultibodyTpl<float>>(robot_model_ptr);
	// crocoddyl::StateMultibody state(robot_model_ptr);
	// crocoddyl::StateMultibodyTpl<float> state_float = state.cast<float>();

	// boost::shared_ptr<crocoddyl::ActuationModelFullTpl<pcg_t>> actuation =
	// 	boost::make_shared<crocoddyl::ActuationModelFullTpl<pcg_t>>(state);

	// // Sample a random configuration
	// Eigen::VectorXf q = randomConfiguration(robot_model);
	// std::cout << "q: " << q.transpose() << std::endl;

	// // Perform the forward kinematics over the kinematic tree
	// forwardKinematics(robot_model, data, q);

	// // Print out the placement of each joint of the kinematic tree
	// for (JointIndex joint_id = 0; joint_id < (JointIndex)robot_model.njoints; ++joint_id)
	// 	std::cout << std::setw(24) << std::left
	// 			  << robot_model.names[joint_id] << ": "
	// 			  << std::fixed << std::setprecision(2)
	// 			  << data.oMi[joint_id].translation().transpose()
	// 			  << std::endl;

	
	// char eePos_traj_file_name[100];
	// char xu_traj_file_name[100];
	// const uint32_t knot_points = KNOT_POINTS;
	// int start_state = 0;
	// int goal_state = 0;

	// snprintf(eePos_traj_file_name, sizeof(eePos_traj_file_name), "testfiles/%d_%d_eepos.traj", start_state, goal_state);
	// std::vector<std::vector<pcg_t>> eePos_traj2d = readCSVToVecVec<pcg_t>(eePos_traj_file_name);

	// snprintf(xu_traj_file_name, sizeof(xu_traj_file_name), "testfiles/%d_%d_traj.csv", start_state, goal_state);
	// std::vector<std::vector<pcg_t>> xu_traj2d = readCSVToVecVec<pcg_t>(xu_traj_file_name);

	// if (eePos_traj2d.size() < knot_points)
	// {
	// 	std::cout << "precomputed traj length < knotpoints, not implemented\n";
	// 	return 0;
	// }

	// std::vector<pcg_t> h_eePos_traj;
	// for (const auto &vec : eePos_traj2d)
	// {
	// 	h_eePos_traj.insert(h_eePos_traj.end(), vec.begin(), vec.end());
	// }
	// std::vector<pcg_t> h_xu_traj;
	// for (const auto &xu_vec : xu_traj2d)
	// {
	// 	h_xu_traj.insert(h_xu_traj.end(), xu_vec.begin(), xu_vec.end());
	// }

	// // Create the x0, assuming x0 is a vector of size 14
	// Eigen::VectorXd x0(NUM_STATES);
	// for (int i = 0; i < NUM_STATES; ++i)
	// {
	// 	x0(i) = h_xu_traj[i];
	// }

	// // // Initialize matrices Q and R for state and control tracking cost
	// Eigen::VectorXf Q = Eigen::VectorXf::Ones(NUM_STATES) * Q_COST;
	// boost::shared_ptr<crocoddyl::ActivationModelWeightedQuadTpl<pcg_t>> activation_model_state = 
	// 	boost::make_shared<crocoddyl::ActivationModelWeightedQuadTpl<pcg_t>>(Q);
	// Eigen::MatrixXf R = Eigen::MatrixXf::Identity(NUM_CONTROLS, NUM_CONTROLS) * 0.0001;
	// boost::shared_ptr<crocoddyl::ActivationModelWeightedQuadTpl<pcg_t>> activation_model_control = 
	// 	boost::make_shared<crocoddyl::ActivationModelWeightedQuadTpl<pcg_t>>(R);


	// // // Initialize the cost models and action models
	// // boost::shared_ptr<crocoddyl::CostModelSumTpl<pcg_t>> running_cost_model =
	// // 	boost::make_shared<crocoddyl::CostModelSumTpl<pcg_t>>(state);
	// // boost::shared_ptr<crocoddyl::CostModelSumTpl<pcg_t>> terminal_cost_model =
	// // 	boost::make_shared<crocoddyl::CostModelSumTpl<pcg_t>>(state);

	// printf("initialized the cost models\n");

	// for (int t = 0; t < KNOT_POINTS; ++t)
	// {
	// 	Eigen::VectorXf x_ref_t(NUM_STATES);
	// 	Eigen::VectorXf u_ref_t(NUM_CONTROLS);
	// 	Eigen::VectorXf ee_pos_target(EE_SIZE);
	// 	// read into x_ref_t and u_ref_t from xu_traj2d
	// 	for (int i = 0; i < NUM_STATES; ++i)
	// 	{
	// 		x_ref_t(i) = xu_traj2d[t][i];
	// 		if (i < NUM_CONTROLS) {
	// 			u_ref_t(i) = xu_traj2d[t][i + 14];
	// 		}
	// 		if (i < EE_SIZE) {
	// 			ee_pos_target(i) = eePos_traj2d[t][i];
	// 		}
	// 	}

	// 	// Commenting out adding state cost for now, we don't penalize position in gpu-pcg-tracking, only velocity,
	// 	// and there doesn't seem to be a supported way to do that in crocoddyl
	// 	// boost::shared_ptr<crocoddyl::ResidualModelState> state_residual =
	// 	// 	boost::make_shared<crocoddyl::ResidualModelState>(state, x_ref_t);
	// 	// printf("initialized state_residual\n");
	// 	// boost::shared_ptr<crocoddyl::CostModelResidual> state_cost =
	// 	// 	boost::make_shared<crocoddyl::CostModelResidual>(state, state_residual);
	// 	// printf("initialized state_cost\n");

	// 	// running_cost_model->addCost("stateTrack" + std::to_string(t), state_cost, 1.);

	// 	// printf("Added state cost for knot point %d\n", t);

	// 	if (t < (KNOT_POINTS - 1)) {
	// 		// Add control cost if this is not the final knot point
	// 		boost::shared_ptr<crocoddyl::ResidualModelControlTpl<pcg_t>> control_residual =
	// 			boost::make_shared<crocoddyl::ResidualModelControlTpl<pcg_t>>(state, u_ref_t);
	// 		boost::shared_ptr<crocoddyl::CostModelResidualTpl<pcg_t>> control_cost =
	// 			boost::make_shared<crocoddyl::CostModelResidualTpl<pcg_t>>(state, control_residual);

	// 		running_cost_model->addCost("controlTrack" + std::to_string(t), control_cost, 0.001);

	// 		printf("Added control cost for knot point %d\n", t);
	// 	}
	// 	// else {
	// 	// 	// Add terminal cost if this is final knot point
	// 	// 	boost::shared_ptr<crocoddyl::ResidualModelState> terminal_state_residual =
	// 	// 		boost::make_shared<crocoddyl::ResidualModelState>(state, x_ref_t); // 0: nu
	// 	// 	boost::shared_ptr<crocoddyl::CostModelResidual<float>> terminal_state_cost =
	// 	// 		boost::make_shared<crocoddyl::CostModelResidual<float>>(state, activation_model_state, terminal_state_residual);
			
	// 	// 	terminal_cost_model->addCost("stateTrack" + std::to_string(t), terminal_state_cost, 10000.);

	// 	// 	printf("Added terminal cost for knot point %d\n", t);
	// 	// 	// print out the x_ref_t at this point so we know what the goal is
	// 	// 	std::cout << "x_ref_t: " << x_ref_t << std::endl;
	// 	// }
	// }

	// // Create the action models
	// boost::shared_ptr<crocoddyl::IntegratedActionModelEulerTpl<pcg_t>> running_model =
	// 	boost::make_shared<crocoddyl::IntegratedActionModelEulerTpl<pcg_t>>(
	// 		boost::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamicsTpl<pcg_t>>(state, actuation, running_cost_model),
	// 		DT);
	// boost::shared_ptr<crocoddyl::IntegratedActionModelEulerTpl<pcg_t>> terminal_model =
	// 	boost::make_shared<crocoddyl::IntegratedActionModelEulerTpl<pcg_t>>(
	// 		boost::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamicsTpl<pcg_t>>(state, actuation, terminal_cost_model),
	// 		DT);

	// printf("initialized the action models\n");

	// // Create the problem
	// std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<pcg_t>>> running_models(
	// 	KNOT_POINTS, running_model);
	// crocoddyl::ShootingProblemTpl<pcg_t> problem(x0, running_models, terminal_model);
	// boost::shared_ptr<crocoddyl::ShootingProblem> problem_ptr = boost::make_shared<crocoddyl::ShootingProblem>(x0, running_models, terminal_model);


	// // Create the DDP solver
	// crocoddyl::SolverDDP ddp(problem_ptr);

	// // Set up callback
	// std::vector<boost::shared_ptr<crocoddyl::CallbackAbstract>> cbs;
	// cbs.push_back(boost::make_shared<crocoddyl::CallbackVerbose>());
	// ddp.setCallbacks(cbs);

	// // Solve the problem
	// ddp.solve();

	// //print the initial state x0 for reference
	// std::cout << "Initial state: " << x0 << std::endl;

	// Eigen::VectorXd final_state = ddp.get_xs().back();
    
    // // Print it
    // std::cout << "Final state: " << final_state << std::endl;

	return 0;
}