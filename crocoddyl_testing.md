Hi Carlos,

Thanks for all your work creating and open sourcing such a great trajectory optimization library. I have gone through the tutorials and have been experimenting with it, and have been finding it to work very smoothly. I am able to get a DDP solve working correctly on the IIWA arm, both in python via the bindings and in the c++ directly. However we really want to test with floats, rather than with doubles, and when I try to switch to the templated versions of the functions and pass float as the template parameter, I am running into issues. I saw from ![this](https://github.com/stack-of-tasks/pinocchio/issues/2037) github issue that for parsing a URDF only doubles are supported, but that they can then be casted to floats, which is what I am doing for the robot model I create, and this part works correctly:

```
ModelTpl<double> model_double;
pinocchio::urdf::buildModel(urdf_filename, model_double);
pinocchio::ModelTpl<float> model = model_double.cast<float>();
```

However, I am unable to create a state multibody object using the float robot model, and I am not sure why. What I basically want to do is

```
boost::shared_ptr<crocoddyl::StateMultibodyTpl<float>> state =
    boost::make_shared<crocoddyl::StateMultibodyTpl<float>>(
        boost::make_shared<pinocchio::ModelTpl<float>>(model));
```

But this results in the following errors:

```
/home/a2rlab/anaconda3/envs/crocoddyl/lib/pkgconfig/../../include/crocoddyl/multibody/states/multibody.hxx:64:6:   required from here
/home/a2rlab/anaconda3/envs/crocoddyl/lib/pkgconfig/../../include/pinocchio/multibody/liegroup/special-orthogonal.hpp:74:45: error: no matching function for call to ‘if_then_else(pinocchio::internal::ComparisonOperators, const Scalar&, double, float, pinocchio::internal::if_then_else_impl<float, float, float, float>::ReturnType)’
   74 |                                 if_then_else(internal::GT, tr, Scalar(2) - 1e-2,
      |                                 ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   75 |                                              asin((R(1,0) - R(0,1)) / Scalar(2)), // then
      |                                              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   76 |                                              if_then_else(internal::GE, R (1, 0), Scalar(0),
      |                                              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   77 |                                                           acos(tr/Scalar(2)), // then
      |                                                           ~~~~~~~~~~~~~~~~~~~~~~~~~~~
   78 |                                                           -acos(tr/Scalar(2))
      |                                                           ~~~~~~~~~~~~~~~~~~~
   79 |                                                           )
      |                                                           ~
   80 |                                              )
      |                                              ~
In file included from /home/a2rlab/anaconda3/envs/crocoddyl/lib/pkgconfig/../../include/pinocchio/math/quaternion.hpp:16,
                 from /home/a2rlab/anaconda3/envs/crocoddyl/lib/pkgconfig/../../include/pinocchio/spatial/se3-tpl.hpp:12,
                 from /home/a2rlab/anaconda3/envs/crocoddyl/lib/pkgconfig/../../include/pinocchio/spatial/se3.hpp:44,
                 from /home/a2rlab/anaconda3/envs/crocoddyl/lib/pkgconfig/../../include/pinocchio/multibody/model.hpp:10,
                 from /home/a2rlab/anaconda3/envs/crocoddyl/lib/pkgconfig/../../include/pinocchio/parsers/urdf.hpp:9,
                 from test_croc.cpp:1:
/home/a2rlab/anaconda3/envs/crocoddyl/lib/pkgconfig/../../include/pinocchio/utils/static-if.hpp:77:5: note: candidate: ‘template<class LhsType, class RhsType, class ThenType, class ElseType> typename pinocchio::internal::if_then_else_impl<LhsType, RhsType, ThenType, ElseType>::ReturnType pinocchio::internal::if_then_else(pinocchio::internal::ComparisonOperators, const LhsType&, const RhsType&, const ThenType&, const ElseType&)’
   77 |     if_then_else(const ComparisonOperators op,
      |     ^~~~~~~~~~~~
/home/a2rlab/anaconda3/envs/crocoddyl/lib/pkgconfig/../../include/pinocchio/utils/static-if.hpp:77:5: note:   template argument deduction/substitution failed:

...

home/a2rlab/anaconda3/envs/crocoddyl/lib/pkgconfig/../../include/crocoddyl/multibody/states/multibody.hxx:64:6:   required from here
/home/a2rlab/anaconda3/envs/crocoddyl/lib/pkgconfig/../../include/pinocchio/utils/static-if.hpp:77:5: error: invalid use of incomplete type ‘struct pinocchio::internal::if_then_else_impl<float, double, float, float>’

```

The ouput is very large so I have truncated it and provided what seem like the most relevant snippets. The full output I'll attach as a separate file in case that is helpful though.

This construction is the same thing I see done in the ![arm.hpp benchmark here](https://github.com/loco-3d/crocoddyl/blob/3de0336178e09b4296912769b494f5741705775f/benchmark/factory/arm.hpp#L70). And to be sure I was doing the same thing, I tried calling this build_arm_action_models function directly in an extremely simple program:

```
#include <iostream>
#define EXAMPLE_ROBOT_DATA_MODEL_DIR "/home/a2rlab/anaconda3/envs/crocoddyl/share/example-robot-data/robots"
#include "crocoddyl/benchmark/factory/arm.hpp"

int main() {
    boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<double>> runningModel;
    boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<double>> terminalModel;
    crocoddyl::benchmark::build_arm_action_models<double>(runningModel, terminalModel);

    if (runningModel && terminalModel) {
        std::cout << "Done" << std::endl;
    }
    return 0;
}
```
This compiles and runs successfully. However, when I try to change the template parameter to float, I get the following errors:

```
required from here
/usr/include/eigen3/Eigen/src/Core/MatrixBase.h:75:17: error: ‘coeff’ has not been declared in ‘Eigen::MatrixBase<Eigen::Product<Eigen::Product<Eigen::Transpose<const Eigen::Block<Eigen::Matrix<float, -1, -1>, -1, -1, false> >, Eigen::DiagonalWrapper<const Eigen::Block<Eigen::Matrix<double, -1, 1>, -1, 1, false> >, 1>, Eigen::Block<Eigen::Matrix<float, -1, -1>, -1, -1, false>, 0> >::Base’
   75 |     using Base::coeff;
      |                 ^~~~~
/usr/include/eigen3/Eigen/src/Core/MatrixBase.h:78:17: error: ‘eval’ has not been declared in ‘Eigen::MatrixBase<Eigen::Product<Eigen::Product<Eigen::Transpose<const Eigen::Block<Eigen::Matrix<float, -1, -1>, -1, -1, false> >, Eigen::DiagonalWrapper<const Eigen::Block<Eigen::Matrix<double, -1, 1>, -1, 1, false> >, 1>, Eigen::Block<Eigen::Matrix<float, -1, -1>, -1, -1, false>, 0> >::Base’
   78 |     using Base::eval;
      |                 ^~~~
/usr/include/eigen3/Eigen/src/Core/MatrixBase.h:79:25: error: ‘operator-’ has not been declared in ‘Eigen::MatrixBase<Eigen::Product<Eigen::Product<Eigen::Transpose<const Eigen::Block<Eigen::Matrix<float, -1, -1>, -1, -1, false> >, Eigen::DiagonalWrapper<const Eigen::Block<Eigen::Matrix<double, -1, 1>, -1, 1, false> >, 1>, Eigen::Block<Eigen::Matrix<float, -1, -1>, -1, -1, false>, 0> >::Base’
   79 |     using Base::operator-;
      |                         ^
/usr/include/eigen3/Eigen/src/Core/MatrixBase.h:82:25: error: ‘operator*=’ has not been declared in ‘Eigen::MatrixBase<Eigen::Product<Eigen::Product<Eigen::Transpose<const Eigen::Block<Eigen::Matrix<float, -1, -1>, -1, -1, false> >, Eigen::DiagonalWrapper<const Eigen::Block<Eigen::Matrix<double, -1, 1>, -1, 1, false> >, 1>, Eigen::Block<Eigen::Matrix<float, -1, -1>, -1, -1, false>, 0> >::Base’
   82 |     using Base::operator*=;
      |                         ^~
/usr/include/eigen3/Eigen/src/Core/MatrixBase.h:83:25: error: ‘operator/=’ has not been declared in ‘Eigen::MatrixBase<Eigen::Product<Eigen::Product<Eigen::Transpose<const Eigen::Block<Eigen::Matrix<float, -1, -1>, -1, -1, false> >, Eigen::DiagonalWrapper<const Eigen::Block<Eigen::Matrix<double, -1, 1>, -1, 1, false> >, 1>, Eigen::Block<Eigen::Matrix<float, -1, -1>, -1, -1, false>, 0> >::Base’
   83 |     using Base::operator/=;
      |                         ^~

```

And also:

```
/home/a2rlab/anaconda3/envs/crocoddyl/lib/pkgconfig/../../include/crocoddyl/multibody/states/multibody.hxx:64:6:   required from here
/home/a2rlab/anaconda3/envs/crocoddyl/lib/pkgconfig/../../include/pinocchio/multibody/liegroup/special-orthogonal.hpp:74:45: error: no matching function for call to ‘if_then_else(pinocchio::internal::ComparisonOperators, const Scalar&, double, float, pinocchio::internal::if_then_else_impl<float, float, float, float>::ReturnType)’
   74 |                                 if_then_else(internal::GT, tr, Scalar(2) - 1e-2,
      |                                 ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   75 |                                              asin((R(1,0) - R(0,1)) / Scalar(2)), // then
      |                                              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   76 |                                              if_then_else(internal::GE, R (1, 0), Scalar(0),
      |                                              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   77 |                                                           acos(tr/Scalar(2)), // then
      |                                                           ~~~~~~~~~~~~~~~~~~~~~~~~~~~
   78 |                                                           -acos(tr/Scalar(2))
      |                                                           ~~~~~~~~~~~~~~~~~~~
   79 |                                                           )
      |                                                           ~
   80 |                                              )
      |                                              ~
In file included from /home/a2rlab/anaconda3/envs/crocoddyl/lib/pkgconfig/../../include/pinocchio/math/quaternion.hpp:16,
                 from /home/a2rlab/anaconda3/envs/crocoddyl/lib/pkgconfig/../../include/pinocchio/spatial/se3-tpl.hpp:12,
                 from /home/a2rlab/anaconda3/envs/crocoddyl/lib/pkgconfig/../../include/pinocchio/spatial/se3.hpp:44,
                 from /home/a2rlab/anaconda3/envs/crocoddyl/lib/pkgconfig/../../include/pinocchio/multibody/model.hpp:10,
                 from /home/a2rlab/anaconda3/envs/crocoddyl/lib/pkgconfig/../../include/pinocchio/algorithm/model.hpp:8,
                 from crocoddyl/benchmark/factory/arm.hpp:13,
                 from test_arm_benchmark.cpp:5:
/home/a2rlab/anaconda3/envs/crocoddyl/lib/pkgconfig/../../include/pinocchio/utils/static-if.hpp:77:5: note: candidate: ‘template<class LhsType, class RhsType, class ThenType, class ElseType> typename pinocchio::internal::if_then_else_impl<LhsType, RhsType, ThenType, ElseType>::ReturnType pinocchio::internal::if_then_else(pinocchio::internal::ComparisonOperators, const LhsType&, const RhsType&, const ThenType&, const ElseType&)’
   77 |     if_then_else(const ComparisonOperators op,


```
And also:
```
/home/a2rlab/anaconda3/envs/crocoddyl/lib/pkgconfig/../../include/pinocchio/utils/static-if.hpp:77:5: error: invalid use of incomplete type ‘struct pinocchio::internal::if_then_else_impl<float, double, float, float>’
/home/a2rlab/anaconda3/envs/crocoddyl/lib/pkgconfig/../../include/pinocchio/utils/static-if.hpp:18:12: note: declaration of ‘struct pinocchio::internal::if_then_else_impl<float, double, float, float>’
   18 |     struct if_then_else_impl;

```

I am running on Ubuntu 22.04, and installed crocoddyl via conda. I have version 2.0.0 of Crocoddyl, version 2.6.18 of Pinocchio, and version 3.4.0 of Eigen. Any support or suggestions you can provide would be very much appreciated, and if there is any other additional information which would be helpful, please let me know!

As a separate question, something we would really love to do would be to limit the length of time the DDP solver can run for. I see that we can set the max iters, and also it appears we can change the exit tolerance. There does not appear to be any setting which results in a timeout for the solver. I just was hoping to double check with you that I am not missing anything - is this accurate?

Thank you very much for your time reviewing this, and again for all your work on this library!

Will