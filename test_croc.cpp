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

#include <crocoddyl/multibody/residuals/frame-translation.hpp>

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

#define EE_DIM_POS 3

#define QD_COST 0.1
#define Q_COST 0.1
#define R_COST 0.0001
#define EE_COST 0.5

int main(int argc, char **argv)
{
	using namespace pinocchio;

	// You should change here to set up your own URDF file or just pass it as an argument of this example.
	const std::string urdf_filename = (argc <= 1) ? PINOCCHIO_MODEL_DIR + std::string("/example-robot-data/robots/ur_description/urdf/ur5_robot.urdf") : argv[1];
	// Load the urdf model

	// NOTE: parsing the URDF only works with doubles, if we need to use floats we need to parse first and then cast
	// https://github.com/stack-of-tasks/pinocchio/issues/2037
	pinocchio::ModelTpl<double> model_double;
	// ModelTpl<double> model;
	pinocchio::urdf::buildModel(urdf_filename, model_double);
	// pinocchio::urdf::buildModel(urdf_filename, model);
	pinocchio::ModelTpl<double> model = model_double.cast<double>();

	// // Create data required by the algorithms
	DataTpl<double> data(model);

	// // Create the state and actuation models
	boost::shared_ptr<crocoddyl::StateMultibodyTpl<double>> state =
		boost::make_shared<crocoddyl::StateMultibodyTpl<double>>(
			boost::make_shared<pinocchio::ModelTpl<double>>(model));
	// boost::shared_ptr<pinocchio::ModelTpl<double>> robot_model_ptr = boost::make_shared<pinocchio::ModelTpl<double>>(robot_model);
	// boost::shared_ptr<crocoddyl::StateMultibodyTpl<double>> state = boost::make_shared<crocoddyl::StateMultibodyTpl<double>>(robot_model_ptr);
	// crocoddyl::StateMultibody state(robot_model_ptr);
	// crocoddyl::StateMultibodyTpl<double> state_float = state.cast<double>();

	boost::shared_ptr<crocoddyl::ActuationModelFullTpl<double>> actuation =
		boost::make_shared<crocoddyl::ActuationModelFullTpl<double>>(state);

	// Sample a random configuration
	Eigen::VectorXd q = randomConfiguration(model);
	std::cout << "q: " << q.transpose() << std::endl;

	// Perform the forward kinematics over the kinematic tree
	forwardKinematics(model, data, q);

	// Print out the placement of each joint of the kinematic tree
	for (JointIndex joint_id = 0; joint_id < (JointIndex)model.njoints; ++joint_id)
		std::cout << std::setw(24) << std::left
				  << model.names[joint_id] << ": "
				  << std::fixed << std::setprecision(2)
				  << data.oMi[joint_id].translation().transpose()
				  << std::endl;

	
	char eePos_traj_file_name[100];
	char xu_traj_file_name[100];
	const uint32_t knot_points = KNOT_POINTS;
	int start_state = 0;
	int goal_state = 0;

	snprintf(eePos_traj_file_name, sizeof(eePos_traj_file_name), "testfiles/%d_%d_eepos.traj", start_state, goal_state);
	std::vector<std::vector<double>> eePos_traj2d = readCSVToVecVec<double>(eePos_traj_file_name);

	snprintf(xu_traj_file_name, sizeof(xu_traj_file_name), "testfiles/%d_%d_traj.csv", start_state, goal_state);
	std::vector<std::vector<double>> xu_traj2d = readCSVToVecVec<double>(xu_traj_file_name);

	if (eePos_traj2d.size() < knot_points)
	{
		std::cout << "precomputed traj length < knotpoints, not implemented\n";
		return 0;
	}

	std::vector<double> h_eePos_traj;
	for (const auto &vec : eePos_traj2d)
	{
		h_eePos_traj.insert(h_eePos_traj.end(), vec.begin(), vec.end());
	}
	std::vector<double> h_xu_traj;
	for (const auto &xu_vec : xu_traj2d)
	{
		h_xu_traj.insert(h_xu_traj.end(), xu_vec.begin(), xu_vec.end());
	}

	// Create the x0, assuming x0 is a vector of size 14
	Eigen::VectorXf x0(NUM_STATES);
	for (int i = 0; i < NUM_STATES; ++i)
	{
		x0(i) = h_xu_traj[i];
	}

	// // Initialize matrices Q and R for state and control tracking cost
	Eigen::VectorXd Q_vec(NUM_STATES);
	Q_vec.fill(Q_COST);
	
	Eigen::VectorXd R_vec(NUM_CONTROLS);
	R_vec.fill(R_COST);

	Eigen::VectorXd EE_penalty_vec(EE_DIM_POS);
	EE_penalty_vec.fill(EE_COST);
	
	boost::shared_ptr<crocoddyl::ActivationModelWeightedQuadTpl<double>> activation_model_state = 
		boost::make_shared<crocoddyl::ActivationModelWeightedQuadTpl<double>>(Q_vec);
	boost::shared_ptr<crocoddyl::ActivationModelWeightedQuadTpl<double>> activation_model_control = 
		boost::make_shared<crocoddyl::ActivationModelWeightedQuadTpl<double>>(R_vec);


	// Initialize the cost models and action models
	boost::shared_ptr<crocoddyl::CostModelSumTpl<double>> running_cost_model =
		boost::make_shared<crocoddyl::CostModelSumTpl<double>>(state);
	boost::shared_ptr<crocoddyl::CostModelSumTpl<double>> terminal_cost_model =
		boost::make_shared<crocoddyl::CostModelSumTpl<double>>(state);

	printf("initialized the cost models\n");

	for (int t = 0; t < KNOT_POINTS; ++t)
	{
		Eigen::VectorXd x_ref_t(NUM_STATES);
		Eigen::VectorXd u_ref_t(NUM_CONTROLS);
		Eigen::VectorXd ee_pos_target(EE_DIM_POS);
		// read into x_ref_t and u_ref_t from xu_traj2d
		for (int i = 0; i < NUM_STATES; ++i)
		{
			x_ref_t(i) = xu_traj2d[t][i];
			if (i < NUM_CONTROLS) {
				u_ref_t(i) = xu_traj2d[t][i + 14];
			}
			if (i < EE_DIM_POS) {
				ee_pos_target(i) = eePos_traj2d[t][i];
			}
		}

		// Commenting out adding state cost for now, we don't penalize position in gpu-pcg-tracking, only velocity,
		// and there doesn't seem to be a supported way to do that in crocoddyl
		// boost::shared_ptr<crocoddyl::ResidualModelState> state_residual =
		// 	boost::make_shared<crocoddyl::ResidualModelState>(state, x_ref_t);
		// printf("initialized state_residual\n");
		// boost::shared_ptr<crocoddyl::CostModelResidual> state_cost =
		// 	boost::make_shared<crocoddyl::CostModelResidual>(state, state_residual);
		// printf("initialized state_cost\n");

		// running_cost_model->addCost("stateTrack" + std::to_string(t), state_cost, 1.);

		// printf("Added state cost for knot point %d\n", t);
		boost::shared_ptr<crocoddyl::ResidualModelStateTpl<double>> terminal_state_residual =
			boost::make_shared<crocoddyl::ResidualModelStateTpl<double>>(state, x_ref_t); // 0: nu

		boost::shared_ptr<crocoddyl::ActivationModelWeightedQuadTpl<double>> activation_ee = 
			boost::make_shared<crocoddyl::ActivationModelWeightedQuadTpl<double>>(EE_penalty_vec);

		boost::shared_ptr<crocoddyl::ResidualModelFrameTranslationTpl<double>> ee_residual =
			boost::make_shared<crocoddyl::ResidualModelFrameTranslationTpl<double>>(state, model.getFrameId("iiwa_joint_7"), ee_pos_target);
		printf("initialized state_residual\n");
		boost::shared_ptr<crocoddyl::CostModelResidualTpl<double>> goal_tracking_xyz_cost =
			boost::make_shared<crocoddyl::CostModelResidualTpl<double>>(state, activation_ee, ee_residual);
		printf("initialized state_cost\n");

		running_cost_model->addCost("goalTrack" + std::to_string(t), goal_tracking_xyz_cost, 1.);

		if (t < (KNOT_POINTS - 1)) {
			// Add control cost if this is not the final knot point
			boost::shared_ptr<crocoddyl::ResidualModelControlTpl<double>> control_residual =
				boost::make_shared<crocoddyl::ResidualModelControlTpl<double>>(state, u_ref_t);
			boost::shared_ptr<crocoddyl::CostModelResidualTpl<double>> control_cost =
				boost::make_shared<crocoddyl::CostModelResidualTpl<double>>(state, control_residual);

			running_cost_model->addCost("controlTrack" + std::to_string(t), control_cost, 0.001);

			printf("Added control cost for knot point %d\n", t);
		}
		else {
			// Add terminal cost if this is final knot point
			// boost::shared_ptr<crocoddyl::ResidualModelStateTpl<double>> terminal_state_residual =
			// 	boost::make_shared<crocoddyl::ResidualModelStateTpl<double>>(state, x_ref_t); // 0: nu
			// boost::shared_ptr<crocoddyl::CostModelResidualTpl<double>> terminal_state_cost =
			// 	boost::make_shared<crocoddyl::CostModelResidualTpl<double>>(state, activation_model_state, terminal_state_residual);
			
			// terminal_cost_model->addCost("stateTrack" + std::to_string(t), terminal_state_cost, 10000.);

			// printf("Added terminal cost for knot point %d\n", t);
			// // print out the x_ref_t at this point so we know what the goal is
			// std::cout << "x_ref_t: " << x_ref_t << std::endl;
		}
	}

	// // Create the action models
	// boost::shared_ptr<crocoddyl::IntegratedActionModelEulerTpl<double>> running_model =
	// 	boost::make_shared<crocoddyl::IntegratedActionModelEulerTpl<double>>(
	// 		boost::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamicsTpl<double>>(state, actuation, running_cost_model),
	// 		DT);
	// boost::shared_ptr<crocoddyl::IntegratedActionModelEulerTpl<double>> terminal_model =
	// 	boost::make_shared<crocoddyl::IntegratedActionModelEulerTpl<double>>(
	// 		boost::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamicsTpl<double>>(state, actuation, terminal_cost_model),
	// 		DT);

	// printf("initialized the action models\n");

	// // Create the problem
	// std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<double>>> running_models(
	// 	KNOT_POINTS, running_model);
	// crocoddyl::ShootingProblemTpl<double> problem(x0, running_models, terminal_model);
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

	// Eigen::VectorXf final_state = ddp.get_xs().back();
    
    // // Print it
    // std::cout << "Final state: " << final_state << std::endl;

	return 0;
}