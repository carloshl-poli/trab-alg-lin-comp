#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

//row-major macro
#define A(i,j) A[i*n + j]


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
    for (int i = 1; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < i; j++) {
            sum += L[i][j] * y[j];
        }
        y[i] = (b[i] - sum) / L[i][i];
    }

    // Fase 3: resolução de Ux = y
    double* x = (double *)malloc(n * sizeof(double));
    for (int i = n - 1; i >= 0; i--) {
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

int main() {
    size_t n = 3;

    // Alocação da matriz A
    double** A = (double**)malloc(n * sizeof(double*));
    for (size_t i = 0; i < n; i++) {
        A[i] = (double*)malloc(n * sizeof(double));
    }

    // Alocação do vetor b
    double* b = (double*)malloc(n * sizeof(double));

    // Preenchendo A
    A[0][0] = 2; A[0][1] = 1; A[0][2] = -1;
    A[1][0] = -3; A[1][1] = -1; A[1][2] = 2;
    A[2][0] = -2; A[2][1] = 1; A[2][2] = 2;

    // Preenchendo b
    b[0] = 8;
    b[1] = -11;
    b[2] = -3;

    // Criando cópias de A e b
    double** A_copy = (double**)malloc(n * sizeof(double*));
    for (size_t i = 0; i < n; i++) {
        A_copy[i] = (double*)malloc(n * sizeof(double));
        for (size_t j = 0; j < n; j++) {
            A_copy[i][j] = A[i][j];
        }
    }

    double* b_copy = (double*)malloc(n * sizeof(double));
    for (size_t i = 0; i < n; i++) {
        b_copy[i] = b[i];
    }

    // Solucionando o sistema
    double* x = gauss_elim(A_copy, b_copy, n);

    if (x == NULL) {
        printf("Sistema singular ou mal condicionado.\n");
    } else {
        printf("Solução:\n");
        for (size_t i = 0; i < n; i++) {
            printf("x[%d] = %.6f\n", i, x[i]);
        }

        // Verificação do resíduo
        double res = compute_residual(A_copy, x, b_copy, n);
        printf("Norma do resíduo: %.6e\n", res);
    }

    // Liberação de memória
    for (size_t i = 0; i < n; i++) {
        free(A[i]);
        free(A_copy[i]);
    }
    free(A);
    free(A_copy);
    free(b);
    free(b_copy);
    free(x);
    printf("program finished\n");

    return 0;
}

