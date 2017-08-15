#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <random>
#include "model.hpp"
//------------------------------------------------------------------------
int
test_baker(Model &model) {
  static std::mt19937 mt;
  std::uniform_real_distribution<float> ud(0.0, 1.0);
  vf x;
  float v = ud(mt);
  for (int i = 0; i < model.n_in; i++) {
    x.push_back(v);
    v = v * 3.0;
    v = v - int(v);
  }
  return model.argmax(x);
}
//------------------------------------------------------------------------
int
test_random(Model &model) {
  static std::mt19937 mt;
  std::uniform_real_distribution<float> ud(0.0, 1.0);
  vf x;
  for (int i = 0; i < model.n_in; i++) {
    x.push_back(ud(mt));
  }
  return model.argmax(x);
}
//------------------------------------------------------------------------
void
test(Model &model) {
  vf x;
  for (int i = 0; i < model.n_in; i++) {
    x.push_back(0.5);
  }
  vf y = model.predict(x);
  printf("%f %f\n", y[0], y[1]);
}
//------------------------------------------------------------------------
int
main(void) {
  const int n_in = 200;
  const int n_units = 200;
  const int n_out = 2;
  Model model(n_in, n_units, n_out);
  model.load("baker.dat");
  test(model);
  const int TOTAL = 1000;
  int bn = 0;
  for (int i = 0; i < TOTAL; i++) {
    bn += test_baker(model);
  }
  std::cout << "Check Baker" << std::endl;
  std::cout << "Success/Fail:" << (TOTAL - bn) << "/" << bn << std::endl;
  int rn = 0;
  for (int i = 0; i < TOTAL; i++) {
    rn += test_random(model);
  }
  std::cout << "Check Random" << std::endl;
  std::cout << "Success/Fail:" << rn << "/" << (TOTAL - rn) << std::endl;
}
//------------------------------------------------------------------------
