#add_executable(cylindrical_euler cylindrical_euler/main.cc)
add_executable(khi cylindrical_euler/kelvin_helmholtz.cc)
deal_ii_setup_target(khi)

