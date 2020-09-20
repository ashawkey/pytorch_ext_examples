#include <pybind11/pybind11.h>

#include "src/add.hpp"


/*
 * macro PYBIND11_MODULE(module_name, m)
 *     module_name: import name in python. No Quotes !!!
 *     m: pybind11::module interface
 *         def(name, ptr, description)
 *             name: method name python.
 * */
PYBIND11_MODULE(_backend, m) {
    m.def("forward", &Add_forward_cpu, "Add forward");
    m.def("backward", &Add_backward_cpu, "Add backward");

}
