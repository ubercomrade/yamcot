#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "core_functions.h"

namespace nb = nanobind;

NB_MODULE(_core, m) {
    m.doc() = "C++ backend for yamcot";
    
    m.def("run_motali_cpp", &run_motali_cpp,
          "Run motali comparison algorithm",
          nb::arg("file_fasta"),
          nb::arg("type_model_1"), 
          nb::arg("type_model_2"),
          nb::arg("file_model_1"),
          nb::arg("file_model_2"), 
          nb::arg("file_table_1"),
          nb::arg("file_table_2"),
          nb::arg("shift"),
          nb::arg("pvalue"),
          nb::arg("file_hist"),
          nb::arg("yes_out_hist"),
          nb::arg("file_prc"),
          nb::arg("yes_out_prc"),
          nb::arg("file_short_over"),
          nb::arg("file_short_all"),
          nb::arg("file_sta_long"));
}