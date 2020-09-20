#include <pybind11/pybind11.h>

#include "src/add.hpp"


/*
 * macro PYBIND11_MODULE(module_name, m)
 *     module_name: should not be in quotes. import this name in python.
 *     m: pybind11::module interface
 * */
PYBIND11_MODULE(add_cpp, m) {
    m.def("forward", &Add_forward_cpu, "Add forward");
    m.def("backward", &Add_backward_cpu, "Add backward");

}
