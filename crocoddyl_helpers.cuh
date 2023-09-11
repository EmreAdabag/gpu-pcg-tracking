#include "pinocchio/parsers/urdf.hpp"

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include <Eigen/Dense>
#include <pinocchio/multibody/model.hpp>
#include <crocoddyl/core/actuation/squashing/smooth-sat.hpp>
#include <crocoddyl/core/costs/residual.hpp>
#include <crocoddyl/core/integrator/euler.hpp>
#include <crocoddyl/core/solvers/ddp.hpp>
#include <crocoddyl/core/solvers/fddp.hpp>
#include <crocoddyl/core/utils/callbacks.hpp>
#include <crocoddyl/core/costs/cost-sum.hpp>

#include <crocoddyl/core/activations/weighted-quadratic.hpp>

#include <crocoddyl/multibody/states/multibody.hpp>
#include <crocoddyl/multibody/actuations/full.hpp>
#include <crocoddyl/multibody/residuals/state.hpp>
#include <crocoddyl/core/residuals/control.hpp>
#include <crocoddyl/multibody/actions/free-fwddyn.hpp>

#include <crocoddyl/multibody/residuals/frame-translation.hpp>

pinocchio::Model initialize_pinocchio_robot() {
	const std::string urdf_filename = std::string("/home/a2rlab/.cache/robot_descriptions/drake/manipulation/models/iiwa_description/urdf/iiwa14_primitive_collision.urdf");
	pinocchio::Model robot_model;
	pinocchio::urdf::buildModel(urdf_filename, robot_model);

    return robot_model;
}

boost::shared_ptr<crocoddyl::StateMultibody> initialize_crocoddyl_state_multibody(pinocchio::Model robot_model) {
    boost::shared_ptr<pinocchio::ModelTpl<double>> robot_model_ptr = boost::make_shared<pinocchio::ModelTpl<double>>(robot_model);
	boost::shared_ptr<crocoddyl::StateMultibody> state = boost::make_shared<crocoddyl::StateMultibody>(robot_model_ptr);
    return state;
}

boost::shared_ptr<crocoddyl::ActuationModelFull> initialize_crocoddyl_actuation_model_full(boost::shared_ptr<crocoddyl::StateMultibody> state) {
	boost::shared_ptr<crocoddyl::ActuationModelFull> actuation =
		boost::make_shared<crocoddyl::ActuationModelFull>(state);
    return actuation;
}

crocoddyl::SolverDDP setupCrocoddylProblem(uint32_t state_size, uint32_t control_size, uint32_t knot_points, uint32_t ee_pos_size,
                            boost::shared_ptr<crocoddyl::StateMultibody> state, 
                            boost::shared_ptr<crocoddyl::ActuationModelFull> actuation,
                            pcg_t * h_xu, pcg_t * ee_goal_traj,
							Eigen::VectorXd Q_vec, Eigen::VectorXd QF_vec, Eigen::VectorXd R_vec, Eigen::VectorXd EE_penalty_vec,
							const int ee_joint_frame_id, float timestep) {
	boost::shared_ptr<crocoddyl::CostModelSum> running_cost_model =
		boost::make_shared<crocoddyl::CostModelSum>(state);
	boost::shared_ptr<crocoddyl::CostModelSum> terminal_cost_model =
		boost::make_shared<crocoddyl::CostModelSum>(state);

	// printf("initialized the cost models\n");

	for (int t = 0; t < knot_points; ++t)
	{
		Eigen::VectorXd x_ref_t(state_size);
		Eigen::VectorXd u_ref_t(control_size);
		Eigen::VectorXd u_ref_zero = Eigen::VectorXd::Zero(control_size);
        Eigen::VectorXd eePos_ref_t(ee_pos_size);
		// read into x_ref_t and u_ref_t from xu_traj2d
		for (int i = 0; i < state_size; ++i)
		{
			x_ref_t(i) = h_xu[t * (state_size + control_size) + i];
			if (i < control_size) {
				u_ref_t(i) = h_xu[t * (state_size + control_size) + i + state_size];
			}
            if (i < ee_pos_size) {
                // eePos_ref_t(i) = ee_goal_traj[(knot_points-1) * 6 + i];
                eePos_ref_t(i) = ee_goal_traj[t * 6 + i];
            }
		}

		boost::shared_ptr<crocoddyl::ActivationModelWeightedQuad> activation_state = 
			boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(Q_vec);

		boost::shared_ptr<crocoddyl::ResidualModelState> state_residual =
			boost::make_shared<crocoddyl::ResidualModelState>(state);
		// printf("initialized state_residual\n");
		boost::shared_ptr<crocoddyl::CostModelResidual> state_cost =
			boost::make_shared<crocoddyl::CostModelResidual>(state, activation_state, state_residual);
		// printf("initialized state_cost\n");

		running_cost_model->addCost("stateTrack" + std::to_string(t), state_cost, 0.001);

		// printf("Added state cost for knot point %d\n", t);
		
        // boost::shared_ptr<crocoddyl::ActivationModelWeightedQuad> activation_ee = 
		// 	boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(EE_penalty_vec);

		boost::shared_ptr<crocoddyl::ResidualModelFrameTranslation> ee_residual =
			boost::make_shared<crocoddyl::ResidualModelFrameTranslation>(state, ee_joint_frame_id, eePos_ref_t);
		// printf("initialized state_residual\n");
		// boost::shared_ptr<crocoddyl::CostModelResidual> goal_tracking_xyz_cost =
		// 	boost::make_shared<crocoddyl::CostModelResidual>(state, activation_ee, ee_residual);
		// or, using the default activation function:
		boost::shared_ptr<crocoddyl::CostModelResidual> goal_tracking_xyz_cost =
			boost::make_shared<crocoddyl::CostModelResidual>(state, ee_residual);
		// printf("initialized state_cost\n");

		running_cost_model->addCost("goalTrack" + std::to_string(t), goal_tracking_xyz_cost, 10.);

		// printf("Added state cost for knot point %d\n", t);

		if (t == (KNOT_POINTS - 1)) {
			boost::shared_ptr<crocoddyl::ActivationModelWeightedQuad> activation_state_final = 
				boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(QF_vec);

			boost::shared_ptr<crocoddyl::ResidualModelState> state_residual_final =
				boost::make_shared<crocoddyl::ResidualModelState>(state);
			// printf("initialized state_residual\n");
			boost::shared_ptr<crocoddyl::CostModelResidual> state_cost_final =
				boost::make_shared<crocoddyl::CostModelResidual>(state, activation_state_final, state_residual_final);
			
			terminal_cost_model->addCost("terminalStateCost", state_cost_final, 1.);

			// printf("Added terminal cost for knot point %d\n", t);
			// print out the x_ref_t at this point so we know what the goal is
			// std::cout << "x_ref_t: " << x_ref_t << std::endl;
		} else {
			// otherwise add the control costs
			// Question: should we add the control cost for the terminal knot point? This is terminal for this
			// trajectory, but not necessarily going to be terminal from the MPC standpoint
			// // Add control cost
			boost::shared_ptr<crocoddyl::ActivationModelWeightedQuad> activation_control = 
				boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(R_vec);
			
			// boost::shared_ptr<crocoddyl::ResidualModelControl> control_residual =
			// 	boost::make_shared<crocoddyl::ResidualModelControl>(state, u_ref_t);
			boost::shared_ptr<crocoddyl::ResidualModelControl> control_residual =
				boost::make_shared<crocoddyl::ResidualModelControl>(state);

			boost::shared_ptr<crocoddyl::CostModelResidual> control_cost =
				boost::make_shared<crocoddyl::CostModelResidual>(state, activation_control, control_residual);

			running_cost_model->addCost("controlTrack" + std::to_string(t), control_cost, 0.001);

			// printf("Added control cost for knot point %d\n", t);
		}
	}

	// Create the action models
	boost::shared_ptr<crocoddyl::IntegratedActionModelEuler> running_model =
		boost::make_shared<crocoddyl::IntegratedActionModelEuler>(
			boost::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(state, actuation, running_cost_model),
			timestep);
	boost::shared_ptr<crocoddyl::IntegratedActionModelEuler> terminal_model =
		boost::make_shared<crocoddyl::IntegratedActionModelEuler>(
			boost::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(state, actuation, terminal_cost_model),
			timestep);

	// printf("initialized the action models\n");

	// Create the problem
	std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>> running_models(
		knot_points - 1, running_model);

	Eigen::VectorXd x0(state_size);
	std::copy(h_xu, h_xu + 14, x0.data());

	crocoddyl::ShootingProblem problem(x0, running_models, terminal_model);
	boost::shared_ptr<crocoddyl::ShootingProblem> problem_ptr = boost::make_shared<crocoddyl::ShootingProblem>(x0, running_models, terminal_model);
	// note: I actually think setting the CROCODDYL_WITH_NTHREADS option during compilation is sufficient, verify
	problem_ptr->set_nthreads(CROCODDYL_WITH_NTHREADS);

	// Create the DDP solver
	crocoddyl::SolverDDP ddp(problem_ptr);

    return ddp;
}

template <typename T>
auto crocoddylSolve(uint32_t state_size, uint32_t control_size, uint32_t ee_pos_size,
            uint32_t knot_points, float timestep, T * h_ee_goal_traj, 
			Eigen::VectorXd Q_vec, Eigen::VectorXd QF_vec, Eigen::VectorXd R_vec, Eigen::VectorXd EE_penalty_vec,
            boost::shared_ptr<crocoddyl::StateMultibody> state,
            boost::shared_ptr<crocoddyl::ActuationModelFull> actuation, const int ee_joint_frame_id,
			T * h_xu, int step){

    // ddp timing
    struct timespec ddp_solve_start, ddp_solve_end;
	// if (step == 0) {
	// 	// print the h_xu at the beginning of the solve
	// 	std::cout << "h_xu before: " << std::endl;
	// 	for (int i = 0; i < knot_points; ++i) {
	// 		for (int j = 0; j < state_size; ++j) {
	// 			std::cout << h_xu[i*(state_size+control_size) + j] << " ";
	// 		}
	// 		if (i < knot_points - 1) {
	// 			for (int j = 0; j < control_size; ++j) {
	// 				std::cout << h_xu[i*(state_size+control_size) + j + state_size] << " ";
	// 			}
	// 		}
	// 		std::cout << std::endl;
	// 	}
	// }
    // Question : the regularization techniques are going to be different aren't they? So we 
    // shouldn't need to worry about matching the rho values, we can use the defaults?
    // if(step==0) {
	// 	std::cout << "R_vec: " << std::endl;
	// 	for (int i=0; i < control_size; i++){
	// 		std::cout << R_vec[i] << " ";
	// 	}
	//    std::cout << std::endl;
	
	// 	std::cout << "Q_vec: " << std::endl;
	// 	for( int i=0; i< state_size; i++){
	// 		std::cout << Q_vec[i] << " ";
	// 	}

	// 	std::cout << std::endl;

	// 	std::cout << "QF_vec: " << std::endl;

	// 	for( int i=0; i< state_size; i++){
	// 		std::cout << QF_vec[i] << " ";
	// 	}

	// 	std::cout << std::endl;

	// 	std::cout << "EE_penalty_vec: " << std::endl;

	// 	for( int i=0; i< ee_pos_size; i++){
	// 		std::cout << EE_penalty_vec[i] << " ";
	// 	}

	// 	std::cout << std::endl;
	   
	// }
	// TODO: for sqp we give it an initial position, but we don't give it a full initial trajectory
    crocoddyl::SolverDDP ddp = setupCrocoddylProblem(state_size, control_size, knot_points, 
                    ee_pos_size, state, actuation, h_xu, h_ee_goal_traj, 
					Q_vec, QF_vec, R_vec, EE_penalty_vec, ee_joint_frame_id, timestep);
	
    // Set up callback
	// std::vector<boost::shared_ptr<crocoddyl::CallbackAbstract>> cbs;
	// cbs.push_back(boost::make_shared<crocoddyl::CallbackVerbose>());
	// ddp.setCallbacks(cbs);

	// initialize xs and us from h_xu
	// each 21 elements is a knot point, the first 14 are the state, the next 7 are the control

	// if(step == 0) {

	// 	std::cout<<"State after setup"<<std::endl;
	// 	std::cout << "R_vec: " << std::endl;
	// 	for (int i=0; i < control_size; i++){
	// 		std::cout << R_vec[i] << " ";
	// 	}
	//    std::cout << std::endl;
	
	// 	std::cout << "Q_vec: " << std::endl;
	// 	for( int i=0; i< state_size; i++){
	// 		std::cout << Q_vec[i] << " ";
	// 	}

	// 	std::cout << std::endl;

	// 	std::cout << "QF_vec: " << std::endl;

	// 	for( int i=0; i< state_size; i++){
	// 		std::cout << QF_vec[i] << " ";
	// 	}

	// 	std::cout << std::endl;

	// 	std::cout << "EE_penalty_vec: " << std::endl;

	// 	for( int i=0; i< EE_DIM_POS; i++){
	// 		std::cout << EE_penalty_vec[i] << " ";
	// 	}

	// 	std::cout << std::endl;
	   
	// }
	std::vector<Eigen::VectorXd> x_init;
	std::vector<Eigen::VectorXd> u_init;
	for (int i = 0; i < knot_points; ++i)
	{
		Eigen::VectorXd x_init_i(state_size);
		Eigen::VectorXd u_init_i(control_size);
		for (int j = 0; j < state_size; ++j)
		{
			x_init_i(j) = h_xu[i * (state_size + control_size) + j];
			if (j < control_size && i < knot_points - 1) {
				u_init_i(j) = h_xu[i * (state_size + control_size) + j + state_size];
			}
		}
		
		x_init.push_back(x_init_i);
		
		if (i < knot_points - 1) {
			u_init.push_back(u_init_i);
		}
	}

	// if (step == 0) {
	// 	// print x init vector, knot points rows and state_size columns
		
	// 	std::cout << "x_init: " << std::endl;
	// 	for (int i = 0; i < knot_points; ++i)
	// 	{
	// 		for (int j = 0; j < state_size; ++j)
	// 		{
	// 			std::cout << x_init[i](j) << " ";
	// 		}
	// 		std::cout << std::endl;
	// 	}

	// 	// print u init vector, knot points rows and control_size columns
	// 	std::cout << "u_init: " << std::endl;
	// 	for (int i = 0; i < knot_points - 1; ++i)
	// 	{
	// 		for (int j = 0; j < control_size; ++j)
	// 		{
	// 			std::cout << u_init[i](j) << " ";
	// 		}
	// 		std::cout << std::endl;
	// 	}

	// 	// print the goal trajectory, knot points rows and ee_pos_size columns
	// 	std::cout << "ee_goal_traj: " << std::endl;
	// 	for (int i = 0; i < knot_points; ++i)
	// 	{
	// 		for (int j = 0; j < ee_pos_size; ++j)
	// 		{
	// 			std::cout << h_ee_goal_traj[i * 6 + j] << " ";
	// 		}
	// 		std::cout << std::endl;
	// 	}

	// }
	
	// printf("starting ddp solve, looping over same problem 100000 times\n");
	// for (int counter = 0 ; counter < 100000; counter++) {
	// Solve the problem
	clock_gettime(CLOCK_MONOTONIC, &ddp_solve_start);
	ddp.solve(x_init, u_init, 100, false);
	clock_gettime(CLOCK_MONOTONIC, &ddp_solve_end);

	//print the initial state x0 for reference
	Eigen::VectorXd start_state = ddp.get_xs().front();
	// std::cout << "Initial state: " << start_state << std::endl;

	Eigen::VectorXd final_state = ddp.get_xs().back();
	
	// Print it
	// std::cout << "Final state: " << final_state << std::endl;

	// print the control trajectory in a human readable format
	// std::cout << "Control trajectory: " << std::endl;
	// for (int i = 0; i < knot_points-1; ++i)
	// {
	// 	std::cout << "knot point " << i << ": ";
	// 	for (int j = 0; j < control_size; ++j)
	// 	{
	// 		std::cout << ddp.get_us()[i][j] << " ";
	// 	}
	// 	std::cout << std::endl;
	// }

	// Copy the final trajectory into the output array h_xu
	// the first 14 elements are the state, the next 7 are the control
	// the state is in ddp.xs, the control is in ddp.us
	for (int i = 0; i < knot_points; ++i)
	{
		for (int j = 0; j < state_size; ++j) {
			h_xu[i*(state_size+control_size) + j] = ddp.get_xs()[i][j];
		}
		if (i < knot_points - 1) {
			for (int j = 0; j < control_size; ++j) {
				h_xu[i*(state_size+control_size) + j + state_size] = ddp.get_us()[i][j];
			}
		}
	}

	// if (step == 0) {
	// 	// print the h_xu we are returning from the solve
	// 	std::cout << "h_xu: " << std::endl;
	// 	for (int i = 0; i < knot_points; ++i) {
	// 		for (int j = 0; j < state_size; ++j) {
	// 			std::cout << h_xu[i*(state_size+control_size) + j] << " ";
	// 		}
	// 		if (i < knot_points - 1) {
	// 			for (int j = 0; j < control_size; ++j) {
	// 				std::cout << h_xu[i*(state_size+control_size) + j + state_size] << " ";
	// 			}
	// 		}
	// 		std::cout << std::endl;
	// 	}
	// }

	double ddp_solve_time = time_delta_us_timespec(ddp_solve_start, ddp_solve_end);

	double total_cost = ddp.get_cost();
	// print ddp cost
	// std::cout << "DDP cost: " << total_cost << std::endl;
	// }

    return std::make_tuple(time_delta_us_timespec(ddp_solve_start, ddp_solve_end), ddp.get_iter(), ddp.get_cost());
}