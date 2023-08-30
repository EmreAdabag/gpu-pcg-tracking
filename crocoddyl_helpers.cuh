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

boost::shared_ptr<crocoddyl::ActuationModelFull> initialize_crocoddyl_actuation_model_full(crocoddyl::StateMultibody state) {
	boost::shared_ptr<crocoddyl::ActuationModelFull> actuation =
		boost::make_shared<crocoddyl::ActuationModelFull>(state);
    return actuation;
}

crocoddyl::SolverDDP setupCrocoddylProblem(uint32_t state_size, uint32_t control_size, uint32_t knot_points, uint32_t ee_pos_size,
                            boost::shared_ptr<crocoddyl::StateMultibody> state, 
                            boost::shared_ptr<crocoddyl::ActuationModelFull> actuation,
                            std::vector<std::vector<pcg_t>> xu_traj2d, std::vector<std::vector<pcg_t>> eePos_traj2d,
							Eigen::VectorXd Q_vec, Eigen::VectorXd R_vec, Eigen::VectorXd EE_penalty_vec) {
	boost::shared_ptr<crocoddyl::CostModelSum> running_cost_model =
		boost::make_shared<crocoddyl::CostModelSum>(state);
	boost::shared_ptr<crocoddyl::CostModelSum> terminal_cost_model =
		boost::make_shared<crocoddyl::CostModelSum>(state);

	printf("initialized the cost models\n");

	for (int t = 0; t < knot_points; ++t)
	{
		Eigen::VectorXd x_ref_t(state_size);
		Eigen::VectorXd u_ref_t(control_size);
        Eigen::VectorXd eePos_ref_t(ee_pos_size);
		// read into x_ref_t and u_ref_t from xu_traj2d
		for (int i = 0; i < state_size; ++i)
		{
			x_ref_t(i) = xu_traj2d[t][i];
			if (i < control_size) {
				u_ref_t(i) = xu_traj2d[t][i + state_size];
			}
            if (i < ee_pos_size) {
                eePos_ref_t(i) = eePos_traj2d[t][i];
            }
		}

		boost::shared_ptr<crocoddyl::ActivationModelWeightedQuad> activation_state = 
			boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(Q_vec);

		boost::shared_ptr<crocoddyl::ResidualModelState> state_residual =
			boost::make_shared<crocoddyl::ResidualModelState>(state, x_ref_t);
		printf("initialized state_residual\n");
		boost::shared_ptr<crocoddyl::CostModelResidual> state_cost =
			boost::make_shared<crocoddyl::CostModelResidual>(state, activation_state, state_residual);
		printf("initialized state_cost\n");

		running_cost_model->addCost("stateTrack" + std::to_string(t), state_cost, 1.);

		printf("Added state cost for knot point %d\n", t);
		
        boost::shared_ptr<crocoddyl::ActivationModelWeightedQuad> activation_ee = 
			boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(EE_penalty_vec);

		boost::shared_ptr<crocoddyl::ResidualModelFrameTranslation> ee_residual =
			boost::make_shared<crocoddyl::ResidualModelFrameTranslation>(state, robot_model.getFrameId("iiwa_joint_7"), ee_ref_t);
		printf("initialized state_residual\n");
		boost::shared_ptr<crocoddyl::CostModelResidual> goal_tracking_xyz_cost =
			boost::make_shared<crocoddyl::CostModelResidual>(state, activation_ee, ee_residual);
		printf("initialized state_cost\n");

		running_cost_model->addCost("goalTrack" + std::to_string(t), goal_tracking_xyz_cost, 1.);

		printf("Added state cost for knot point %d\n", t);

		if (t == (KNOT_POINTS - 1)) {
			
			terminal_cost_model->addCost("terminalStateCost", state_cost, 10000.);

			printf("Added terminal cost for knot point %d\n", t);
			// print out the x_ref_t at this point so we know what the goal is
			std::cout << "x_ref_t: " << x_ref_t << std::endl;
		} else {
			// otherwise add the control costs
			// Question: should we add the control cost for the terminal knot point? This is terminal for this
			// trajectory, but not necessarily going to be terminal from the MPC standpoint
			// // Add control cost
			boost::shared_ptr<crocoddyl::ActivationModelWeightedQuad> activation_control = 
				boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(R_vec);
			
			boost::shared_ptr<crocoddyl::ResidualModelControl> control_residual =
				boost::make_shared<crocoddyl::ResidualModelControl>(state, u_ref_t);
			boost::shared_ptr<crocoddyl::CostModelResidual> control_cost =
				boost::make_shared<crocoddyl::CostModelResidual>(state, activation_control, control_residual);

			running_cost_model->addCost("controlTrack" + std::to_string(t), control_cost, 0.001);

			printf("Added control cost for knot point %d\n", t);
		}
	}

	// Create the action models
	boost::shared_ptr<crocoddyl::IntegratedActionModelEuler> running_model =
		boost::make_shared<crocoddyl::IntegratedActionModelEuler>(
			boost::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(state, actuation, running_cost_model),
			DT);
	boost::shared_ptr<crocoddyl::IntegratedActionModelEuler> terminal_model =
		boost::make_shared<crocoddyl::IntegratedActionModelEuler>(
			boost::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(state, actuation, terminal_cost_model),
			DT);

	printf("initialized the action models\n");

	// Create the problem
	std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>> running_models(
		KNOT_POINTS, running_model);
	crocoddyl::ShootingProblem problem(x0, running_models, terminal_model);
	boost::shared_ptr<crocoddyl::ShootingProblem> problem_ptr = boost::make_shared<crocoddyl::ShootingProblem>(x0, running_models, terminal_model);

	// Create the DDP solver
	crocoddyl::SolverDDP ddp(problem_ptr);

    return ddp;
}

template <typename T>
auto crocoddylSolve(uint32_t state_size, uint32_t control_size, uint32_t ee_pos_size,
            uint32_t knot_points, float timestep, std::vector<std::vector<pcg_t>> h_eePos_traj, 
            std::vector<std::vector<pcg_t>> h_xu, boost::shared_ptr<crocoddyl::StateMultibody> state,
            boost::shared_ptr<crocoddyl::ActuationModelFull> actuation){


    // data storage
    std::vector<int> ddp_iter_vec;
    std::vector<bool> ddp_exit_vec;

    bool ddp_time_exit = 1;     // for data recording, not a flag

    // ddp timing
    struct timespec ddp_solve_start, ddp_solve_end;
    clock_gettime(CLOCK_MONOTONIC, &ddp_solve_start);
    uint32_t ddp_iter = 0;

    // Question : the regularization techniques are going to be different aren't they? So we 
    // shouldn't need to worry about matching the rho values, we can use the defaults?

    crocoddyl::SolverDDP ddp = setupCrocoddylProblem(state_size, control_size, knot_points, 
                    ee_pos_size, state, actuation, h_xu, h_eePos_traj);
	
    // Set up callback
	std::vector<boost::shared_ptr<crocoddyl::CallbackAbstract>> cbs;
	cbs.push_back(boost::make_shared<crocoddyl::CallbackVerbose>());
	ddp.setCallbacks(cbs);

	// Solve the problem
	ddp.solve();
    clock_gettime(CLOCK_MONOTONIC, &ddp_solve_end);

	//print the initial state x0 for reference
	std::cout << "Initial state: " << x0 << std::endl;

	Eigen::VectorXd final_state = ddp.get_xs().back();
    
    // Print it
    std::cout << "Final state: " << final_state << std::endl;

    double ddp_solve_time = time_delta_us_timespec(ddp_solve_start, ddp_solve_end);

    return std::make_tuple(ddp_solve_time, ddp.get_iter());
}