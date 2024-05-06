#include "python_wrapper.h"

namespace py = pybind11;



PYBIND11_MODULE(accel_sim, m) {
    py::class_<accel_sim_framework>(m, "accel_sim_framework")
        .def(py::init<std::string &, std::string &>())
        .def("init", &accel_sim_framework::init)
        .def("simulation_loop", &accel_sim_framework::simulation_loop)
        .def("parse_commandlist", &accel_sim_framework::parse_commandlist)
        .def("cleanup", &accel_sim_framework::cleanup)
        .def("simulate", &accel_sim_framework::simulate);

}