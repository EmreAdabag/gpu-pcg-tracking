#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>
#include <math.h>


int main() {

  // float epsilon = .0001;
  // float a = 1.30;
  // float b = .1;
  // std::cout << std::setprecision(std::numeric_limits<float>::max_digits10+1) << "a: " << a << " b: " << b << "\n";
  // std::cout << std::setprecision(std::numeric_limits<float>::max_digits10+1) << a/b << "\n";
  // std::cout << "floor " << std::floor(static_cast<float>(a / b + epsilon)) << std::endl;

  for (int i = 0; i < 2; i++){
    std::vector<int> a;
    a.push_back(1);
    std::cout << a.size() << std::endl;
  }


  return 0;
}