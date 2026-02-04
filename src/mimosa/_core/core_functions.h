#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Function declaration for the main computational function
int run_motali_cpp(
    const char* file_fasta,
    const char* type_model_1,
    const char* type_model_2,
    const char* file_model_1,
    const char* file_model_2,
    const char* file_table_1,
    const char* file_table_2,
    int shift,
    double pvalue,
    const char* file_hist,
    int yes_out_hist,
    const char* file_prc,
    int yes_out_prc,
    const char* file_short_over,
    const char* file_short_all,
    const char* file_sta_long
);

#ifdef __cplusplus
}
#endif