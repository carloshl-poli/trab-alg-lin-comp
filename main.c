#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

//row-major macro
#define A(i,j) A[i*n + j]

int getMin(int x, int y) {
    return x < y ? x : y;
}

void swap_rows(double* matrix, double* temp, size_t columns_amount, size_t row_index_a, size_t row_index_b) {
    size_t row_size = columns_amount * sizeof(double);
    double* row_a = matrix + row_index_a * columns_amount;
    double* row_b = matrix + row_index_b * columns_amount;

    memcpy(temp, row_a, row_size);
    memcpy(row_a, row_b, row_size);
    memcpy(row_b, temp, row_size);
}


double* gauss_elim_row_major(double* A, double* b, size_t n) {
    double* temp = (double *)malloc(n * sizeof(double));
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

        swap_rows(A, temp, n, k, max_row);
        double temp_b = b[k];
        b[k] = b[max_row];
        b[max_row] = temp_b;

        for (size_t i = (k+1); i < n; i++) {
            double m = A(i,k) / A(k,k);
            for (size_t j = k; j < n; j++) {
                A(i,j) -= m * A(k,j);
            }
            b[i] = b[i] - m * b[k];
        }
    }

    double* x = temp;
    for (int i = (int)(n - 1); i >= 0; i--) {
            double sum = 0;
            for (size_t j = i + 1; j < n; j++) {
                sum += A(i,j) * x[j];
            }
            x[i] = (b[i] - sum) / A(i,i);
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
    
    // Solving Ly = b
    y[0] = b[0] / L[0][0];
    for (size_t i = 1; i < n; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < i; j++) {
            sum += L[i][j] * y[j];
        }
        y[i] = (b[i] - sum) / L[i][i];
    }

    // Solving Ux = y
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
        if (hasConverged(new_x, past_x, n, tolerance, false, k, max_iter, log_file)) {
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
// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
// Prototypes (certifique-se de que esses estão no seu código principal ou num header)
#define MAX_PATH 128

double** aloca_matriz(size_t n) {
    double** M = (double**)malloc(n * sizeof(double*));
    for (size_t i = 0; i < n; i++) {
        M[i] = (double*)malloc(n * sizeof(double));
    }
    return M;
}

void libera_matriz(double** M, size_t n) {
    for (size_t i = 0; i < n; i++) free(M[i]);
    free(M);
}

double** le_matriz(const char* path, size_t n) {
    FILE* f = fopen(path, "r");
    if (!f) { perror(path); exit(1); }

    double** M = aloca_matriz(n);
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            fscanf(f, "%lf", &M[i][j]);
    fclose(f);
    return M;
}

double* le_vetor(const char* path, size_t n) {
    FILE* f = fopen(path, "r");
    if (!f) { perror(path); exit(1); }

    double* v = (double*)malloc(n * sizeof(double));
    for (size_t i = 0; i < n; i++)
        fscanf(f, "%lf", &v[i]);
    fclose(f);
    return v;
}

void testa_matriz(int id, size_t n, bool usar_gauss, bool usar_lu, bool usar_jacobi) {
    char pathA[MAX_PATH], pathB[MAX_PATH];
    snprintf(pathA, MAX_PATH, "test_subjects/mat%d_A.txt", id);
    snprintf(pathB, MAX_PATH, "test_subjects/mat%d_b.txt", id);

    printf("\n========= Matriz %d (%zu x %zu) =========\n", id, n, n);

    double** A = le_matriz(pathA, n);
    double* b = le_vetor(pathB, n);

    time_t start, end;
    double time_taken;

    if (usar_gauss) {
        double** A_gauss = aloca_matriz(n);
        for (size_t i = 0; i < n; i++) memcpy(A_gauss[i], A[i], n * sizeof(double));
        double* b_gauss = (double*)malloc(n * sizeof(double));
        memcpy(b_gauss, b, n * sizeof(double));
        start = time(NULL);
        double* x = gauss_elim(A_gauss, b_gauss, n);
        end = time(NULL);
        time_taken = difftime(end, start);
        if (x) {
            printf("Method: Gauss Elimination\n");
            printf("Tempo de execução real: %f segundos\n", time_taken);
            getchar();
            for (size_t i = 0; i < n; i++) {
                printf("x[%zu] = %.6f \n", i, x[i]);
            }
            double res = compute_residual(A, x, b, n);
            printf("[Gauss] Resíduo (norma 2) ||Ax - b|| = %.5e\n", res);
            free(x);
        } else {
            printf("[Gauss] Falha na eliminação (pivot zero).\n");
        }
        libera_matriz(A_gauss, n);
        free(b_gauss);
    }

    if (usar_lu) {
        double** U = aloca_matriz(n);
        double** L = (double**)calloc(n, sizeof(double*));
        for (size_t i = 0; i < n; i++) {
            L[i] = (double*)calloc(n, sizeof(double));
            memcpy(U[i], A[i], n * sizeof(double));
        }
        start = time(NULL);
        if (factorization_LU(A, L, U, n) == 0) {
            double* x = solve_factorized_system(L, U, b, n);
            end = time(NULL);
            time_taken = difftime(end, start);
            printf("Method: LU Fatoration\n");
            printf("Tempo de execução real: %f segundos\n", time_taken);
            getchar();
            for (size_t i = 0; i < n; i++) {
                printf("x[%zu] = %.6f \n", i, x[i]);
            }
            double res = compute_residual(A, x, b, n);
            printf("[LU]     Resíduo (norma 2) ||Ax - b|| = %.5e\n", res);
            free(x);
        } else {
            printf("[LU]     Fatoração falhou (pivot zero).\n");
        }
        libera_matriz(L, n);
        libera_matriz(U, n);
    }

    if (usar_jacobi) {
        int max_iter = getMin(1000, n * 15);
        start = time(NULL);
        double* x = gauss_jacobi(A, b, 1e-6, max_iter, n);
        end = time(NULL);
        time_taken = difftime(end, start);
        if (x) {
            printf("Method: Gauss-Jacobi\n");
            printf("Tempo de execução real: %f segundos\n", time_taken);
            getchar();
            for (size_t i = 0; i < n; i++) {
                printf("x[%zu] = %.6f \n", i, x[i]);
            }

            double res = compute_residual(A, x, b, n);
            printf("[Jacobi] Resíduo (norma 2) ||Ax - b|| = %.5e\n", res);
            free(x);
        } else {
            printf("[Jacobi] Falha (divisão por zero ou não convergiu).\n");
        }
    }

    libera_matriz(A, n);
    free(b);
}

// Tabela com configurações: {id, tamanho, gauss, LU, Jacobi}
void roda_todos_os_testes() {
    struct {
        int id; size_t n; bool g, lu, j;
    } testes[] = {
        {1, 3,  true, true, true},
        {2, 5,  true, true, true},
        {3,100, true, true, true},
        {4,500, true, true, true},
        {5, 5,  true, true, false},
        {6, 4,  true, true, true},
        {7, 2,  false,false,true},
        {8,10,  true, true, true},
        {9,10,  true, true, true}
    };

    int escolha;
    char continuar;

    do {
        printf("\nEscolha uma matriz para testar (1 a 9): ");
        if (scanf("%d", &escolha) != 1 || escolha < 1 || escolha > 9) {
            printf("Entrada inválida. Tente novamente.\n");
            while (getchar() != '\n'); // limpa buffer
            continue;
        }

        int index = escolha - 1;
        testa_matriz(
            testes[index].id,
            testes[index].n,
            testes[index].g,
            testes[index].lu,
            testes[index].j
        );

        printf("\nDeseja testar outra matriz? (s/n): ");
        while (getchar() != '\n'); // limpa buffer anterior
        continuar = getchar();

    } while (continuar == 's' || continuar == 'S');
}

int main() {
    roda_todos_os_testes();
    return 0;
}