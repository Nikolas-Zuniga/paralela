#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Función para inicializar una matriz con valores
void init_matrix(double *mat, int rows, int cols) {
    int i;
    for (i = 0; i < rows * cols; i++) {
        mat[i] = (double)rand() / RAND_MAX;
    }
}

// Función para imprimir una matriz (útil para depuración)
void print_matrix(double *mat, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            printf("%f ", mat[i * cols + j]);
        }
        printf("\n");
    }
}

// Función de multiplicación de matriz-vector
void Mat_vect_mult(double *local_A, double *local_x, double *local_y, int local_m, int n, int local_n, MPI_Comm comm) {
    double *x;
    int local_i, j;

    // Asignar memoria para el vector global x
    x = (double *)malloc(n * sizeof(double));

    // Recopilar todas las porciones locales de 'local_x' en un vector global 'x' en cada proceso
    MPI_Allgather(local_x, local_n, MPI_DOUBLE, x, local_n, MPI_DOUBLE, comm);

    // Inicializar el vector de resultado local a 0
    for (local_i = 0; local_i < local_m; local_i++) {
        local_y[local_i] = 0.0;
    }

    // Realizar la multiplicación de matriz-vector
    for (local_i = 0; local_i < local_m; local_i++) {
        for (j = 0; j < n; j++) {
            local_y[local_i] += local_A[local_i * n + j] * x[j];
        }
    }

    // Liberar la memoria asignada para el vector global
    free(x);
}

int main(int argc, char **argv) {
    int rank, size;
    int m = 9;
    int n = 9; // Número de columnas de la matriz global (y tamaño del vector)
    int local_m; // Número de filas locales
    int local_n; // Tamaño del vector local (asumimos que 'n' es divisible por 'size')

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calcular la distribución de filas y el tamaño del vector local
    local_m = m / size;
    local_n = n / size;

    // Asignar memoria para las partes locales
    double *local_A = (double *)malloc(local_m * n * sizeof(double));
    double *local_x = (double *)malloc(local_n * sizeof(double));
    double *local_y = (double *)malloc(local_m * sizeof(double));

    // Inicializar las porciones locales de la matriz y el vector
    init_matrix(local_A, local_m, n);
    init_matrix(local_x, local_n, 1);

    // Llamar a la función de multiplicación
    Mat_vect_mult(local_A, local_x, local_y, local_m, n, local_n, MPI_COMM_WORLD);

    // Recopilar los resultados de todos los procesos en el proceso 0
    if (rank == 0) {
        double *y = (double *)malloc(m * sizeof(double));
        MPI_Gather(local_y, local_m, MPI_DOUBLE, y, local_m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        printf("Vector de resultado final en el proceso 0:\n");
        // No imprimir la matriz para evitar un output muy largo, pero se podría hacer aquí
            for (int i = 0; i < m; i++) {
        printf("%f ", y[i]);
    }
	free(y);
    } else {
        MPI_Gather(local_y, local_m, MPI_DOUBLE, NULL, local_m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Liberar la memoria local
    free(local_A);
    free(local_x);
    free(local_y);

    MPI_Finalize();

    return 0;
}
