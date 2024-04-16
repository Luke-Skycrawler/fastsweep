#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "redistance.h"
#include <pybind11/numpy.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(shaysweep, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: shaysweep

        .. autosummary::
           :toctree: _generate


    )pbdoc";

    py::class_<Redistance>(m, "Redistance")
        .def(py::init<int>())
        .def("redistance", &Redistance::redistance, py::arg("phi"), py::arg("h"), py::arg("init_boundary") = 0)
        .def("redistance_3d", &Redistance::redistance_3d, py::arg("phi"), py::arg("h"), py::arg("init_boundary") = 0)
        .def("redistance_rb", &Redistance::redistance_rb, py::arg("phi"), py::arg("h"), py::arg("init_boundary") = 0);
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
