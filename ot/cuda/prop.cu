/**
 * @file   prop.cu
 * @author Yibo Lin
 * @date   Mar 2020
 */

#include <cstdio>
#include <ot/cuda/prop.cuh>

#define MAX_SPLIT_TRAN 4 
#define MAX_SPLIT 2
#define MAX_TRAN 2

template <typename T>
void print(const T* data, int n, const char* msg) {
    printf("%s[%d] = {", msg, n);
    for (int i = 0; i < n; ++i) {
        printf("%g ", (double)data[i]);
    }
    printf("}\n");
}

void print(const ArcCUDA* data, int n) {
    printf("arcs[%d] = {", n);
    for (int i = 0; i < n; ++i) {
        printf("(%g, %g) ", (double)data[i].from, (double)data[i].to);
    }
    printf("}\n");
}

void print(FlatTableCUDA const& ft) {
    printf("FlatTableCUDA[%u] = {\n", ft.num_tables);
    for (unsigned i = 0; i < ft.num_tables; ++i) {
        printf("slew[%u][%u x %u]\n", i, ft.slew_indices1_start[i + 1] - ft.slew_indices1_start[i], ft.slew_indices2_start[i + 1] - ft.slew_indices2_start[i]);
        printf("ft.slew_indices1: ");
        for (unsigned j = ft.slew_indices1_start[i]; j < ft.slew_indices1_start[i + 1]; ++j) {
            printf("%g ", ft.slew_indices1[j]); 
        }
        printf("\n");
        printf("ft.slew_indices2: ");
        for (unsigned j = ft.slew_indices2_start[i]; j < ft.slew_indices2_start[i + 1]; ++j) {
            printf("%g ", ft.slew_indices2[j]); 
        }
        printf("\n");
        printf("slew table\n");
        for (unsigned j = ft.slew_table_start[i]; j < ft.slew_table_start[i + 1]; ++j) {
            printf("%g ", ft.slew_table[j]); 
            if ((j % (ft.slew_indices2_start[i + 1] - ft.slew_indices2_start[i])) == (ft.slew_indices2_start[i + 1] - ft.slew_indices2_start[i] - 1)) {
                printf("\n");
            }
        }
        printf("\n");
        printf("delay[%u][%u x %u]\n", i, ft.delay_indices1_start[i + 1] - ft.delay_indices1_start[i], ft.delay_indices2_start[i + 1] - ft.delay_indices2_start[i]);
        printf("ft.delay_indices1: ");
        for (unsigned j = ft.delay_indices1_start[i]; j < ft.delay_indices1_start[i + 1]; ++j) {
            printf("%g ", ft.delay_indices1[j]); 
        }
        printf("\n");
        printf("ft.delay_indices2: ");
        for (unsigned j = ft.delay_indices2_start[i]; j < ft.delay_indices2_start[i + 1]; ++j) {
            printf("%g ", ft.delay_indices2[j]); 
        }
        printf("\n");
        printf("delay table\n");
        for (unsigned j = ft.delay_table_start[i]; j < ft.delay_table_start[i + 1]; ++j) {
            printf("%g ", ft.delay_table[j]); 
            if ((j % (ft.delay_indices2_start[i + 1] - ft.delay_indices2_start[i])) == (ft.delay_indices2_start[i + 1] - ft.delay_indices2_start[i] - 1)) {
                printf("\n");
            }
        }
        printf("\n");
    }
    printf("}\n");
}

unsigned PropCUDA::ftid(unsigned arc_id, unsigned el, unsigned irf, unsigned orf) const {
    return arc_id * (MAX_SPLIT_TRAN * MAX_TRAN) + el * (MAX_TRAN * MAX_TRAN)
        + irf * MAX_TRAN + orf; 
}

void prop_cuda(PropCUDA& prop_data_cpu) {
    print(prop_data_cpu.arcs, prop_data_cpu.num_arcs); 
    print(prop_data_cpu.arc_delays, prop_data_cpu.num_arcs, "arc_delays"); 
    print(prop_data_cpu.arc_impulses, prop_data_cpu.num_arcs, "arc_impulses"); 
    print(prop_data_cpu.arc_loads, prop_data_cpu.num_arcs, "arc_loads"); 
    print(prop_data_cpu.arc2ftid, prop_data_cpu.num_arcs, "arc2ftid"); 
    print(prop_data_cpu.ft);
    // kernel propagation  
}
