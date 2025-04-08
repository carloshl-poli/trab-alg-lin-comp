#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h> 

//row-major macro
#define A(i,j) A[i*n + j]

/*
double* gauss_elim_row_major(double* A, double* b, size_t n) {

    // iterate rows
    for (size_t k = 0; k < n; k++) {

        size_t max_row = k;
        for (size_t i = k+1; i < n; i++) {
            if (fabs(A(i,k)) > fabs(A(max_row,k))) {
                max_row = i;
            }
        }

        // Error
        if (fabs(A(max_row,k)) < 1e-12) {
            return NULL;
        }

        double* temp = A[k];
        A[k] = A[max_row];
        A[max_row] = temp;

        double tmp_b = b[k];
        b[k] = b[max_row];
        b[max_row] = tmp_b;

        for (size_t i = (k+1); i < n; i++) {
            double m = A[i][k] / A[k][k];
            for (size_t j = k; j < n; j++) {
                A[i][j] = A[i][j] - m * A[k][j];
            }
            b[i] = b[i] - m * b[k];
        }
    }

    double* x = (double *)malloc(n * sizeof(double));
    for (int i = (int)(n - 1); i >= 0; i--) {
            double sum = 0;
            for (size_t j = i + 1; j < n; j++) {
                sum = sum + A[i][j] * x[j];
            }
            x[i] = (b[i] - sum) / A[i][i];
    }
    return x;
}
*/

// IMPORTANT: 
//  - L must be zero matrix (use calloc instead of malloc to achieve this)
//  - U must be a copy of A          
int factorization_LU(double** A, double** L, double** U, size_t n) {
    for (size_t i = 0; i < n; i++) {
        L[i][i] = 1;
    }

    for (size_t i = 0; i < n; i++) {
        if (fabs(U[i][i]) < 1e-12) return -1;
        double p = U[i][i];
        for (size_t k = i + 1; k < n; k++) {
            double m = U[k][i] / p;  // use U, não A
            L[k][i] = m;             // índice corrigido
            for (size_t j = i; j < n; j++) {
                U[k][j] -= m * U[i][j];  // atualiza U, não A
            } 
        }
    }
    return 0;
}


double* solve_factorized_system(double** L, double** U, double* b, size_t n) {
    double* y = (double *)malloc(n * sizeof(double));
    
    // Fase 2: resolução de Ly = b
    y[0] = b[0] / L[0][0];
    for (size_t i = 1; i < n; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < i; j++) {
            sum += L[i][j] * y[j];
        }
        y[i] = (b[i] - sum) / L[i][i];
    }

    // Fase 3: resolução de Ux = y
    double* x = (double *)malloc(n * sizeof(double));
    for (int i = (int )(n - 1); i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < n; j++) {
            sum += U[i][j] * x[j];
        }
        x[i] = (y[i] - sum) / U[i][i];
    }

    free(y);
    return x;
}





double* gauss_elim(double** A, double* b, size_t n) {

    // Part 1: Building triangular Matrix
    for (size_t k = 0; k < n; k++) {

        // parcial pivoting section
        size_t max_row = k;
        for (size_t i = k+1; i < n; i++) {
            if (fabs(A[i][k]) > fabs(A[max_row][k])) {
                max_row = i;
            }
        }

        // Failed to find a proper pivot
        if (fabs(A[max_row][k]) < 1e-12) {
            return NULL;
        }

        // Line Swap part of pivoting
        // Line Swap of A[][]
        double* temp = A[k];
        A[k] = A[max_row];
        A[max_row] = temp;
        // Line Swap of b[]
        double tmp_b = b[k];
        b[k] = b[max_row];
        b[max_row] = tmp_b;

        // Zero out all the elements below the pivot
        for (size_t i = (k+1); i < n; i++) {
            double m = A[i][k] / A[k][k];
            for (size_t j = k; j < n; j++) {
                A[i][j] = A[i][j] - m * A[k][j];
            }
            b[i] = b[i] - m * b[k];
        }
    }

    // Part 2: finding x[]
    double* x = (double *)malloc(n * sizeof(double));
    for (int i = (int)(n - 1); i >= 0; i--) {
            double sum = 0;
            for (size_t j = i + 1; j < n; j++) {
                sum = sum + A[i][j] * x[j];
            }
            x[i] = (b[i] - sum) / A[i][i];
    }
    return x;
}

bool hasConverged(double* new_x, double* past_x, int n, double tolerance, bool print_residual, size_t current_iter, size_t max_iter, FILE* log_file) {
    double current_residual = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = fabs(new_x[i] - past_x[i]);
        if (diff > current_residual) {
            current_residual = diff;
        }
    }
    if (print_residual) {
        if (current_iter == 1) {
            printf("| %-15s | %-18s | %-18s | %-16s |\n", 
                "Max Iteration", "Current Iteration", "Residual Value", "Tolerance Value");
            if (log_file) {
                fprintf(log_file, "| %-15s | %-18s | %-18s | %-16s |\n", 
                    "Max Iteration", "Current Iteration", "Residual Value", "Tolerance Value");
            }
        }
        printf("| %-15zu | %-18zu | %-18.10f | %-16.10f |\n", 
            max_iter, current_iter, current_residual, tolerance);
        if (log_file) {
            fprintf(log_file, "| %-15zu | %-18zu | %-18.10f | %-16.10f |\n", 
                max_iter, current_iter, current_residual, tolerance);
        }
    }

    return current_residual < tolerance;
}


double* gauss_jacobi(double** A, double* b, double tolerance, size_t max_iter, size_t n) {
    double* past_x = (double *)calloc(n, sizeof(double));
    FILE* log_file = fopen("jacobi_log.txt", "w");
    for (size_t k = 1; k <= max_iter; k++ ) {
        double* new_x = (double *)calloc(n, sizeof(double));
        for (size_t i = 0; i < n; i++) {
            double sum = 0.0;
            for (size_t j = 0; j < n; j++) {
                if (j != i) {
                    sum += A[i][j] * past_x[j];
                }
            }
            if (fabs(A[i][i]) < 1e-14) {
                fprintf(stderr, "Error: Zero division in A[%zu][%zu].\n", i, i);
                free(new_x);
                free(past_x);
                fclose(log_file);
                return NULL;
            }
            new_x[i] = (b[i] - sum) / A[i][i];
        }
        if (hasConverged(new_x, past_x, n, tolerance, true, k, max_iter, log_file)) {
            free(past_x);
            fclose(log_file);
            return new_x;
        }
        memcpy(past_x, new_x, n * sizeof(double));
        free(new_x);
        
    }
    fclose(log_file);
    return past_x;
}

// Função para calcular o resíduo ||Ax - b|| (norma 2)
double compute_residual(double** A, double* x, double* b, size_t n) {
    double norm = 0.0;
    for (size_t i = 0; i < n; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < n; j++) {
            sum += A[i][j] * x[j];
        }
        norm += pow(sum - b[i], 2);
    }
    return sqrt(norm);
}

// Prototypes (certifique-se de que esses estão no seu código principal ou num header)

int main() {
    size_t n = 3;
    double tolerance = 1e-10;
    size_t max_iter = 100;

    // Alocação da matriz A
    double** A = (double**)malloc(n * sizeof(double*));
    for (size_t i = 0; i < n; i++) {
        A[i] = (double*)malloc(n * sizeof(double));
    }

    // Alocação do vetor b
    double* b = (double*)malloc(n * sizeof(double));

    // Sistema: 10x + y + z = 12
    //          x + 10y + z = 12
    //          x + y + 10z = 12

    A[0][0] = 10; A[0][1] = 1;  A[0][2] = 1;
    A[1][0] = 1;  A[1][1] = 10; A[1][2] = 1;
    A[2][0] = 1;  A[2][1] = 1;  A[2][2] = 10;

    b[0] = 12;
    b[1] = 12;
    b[2] = 12;

    // Chamada do método de Jacobi
    double* x = gauss_jacobi(A, b, tolerance, max_iter, n);

    if (x == NULL) {
        printf("Erro ao resolver o sistema com Jacobi.\n");
    } else {
        printf("\nSolução aproximada com Jacobi:\n");
        for (size_t i = 0; i < n; i++) {
            printf("x[%zu] = %.10f\n", i, x[i]);
        }

        double res = compute_residual(A, x, b, n);
        printf("Norma do resíduo: %.10e\n", res);

        free(x);
    }

    // Liberação de memória
    for (size_t i = 0; i < n; i++) {
        free(A[i]);
    }
    free(A);
    free(b);

    printf("program finished\n");
    return 0;
}

