#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void llenar_matriz(double *matriz, int filas, int columnas) {
    for (int i = 0; i < filas * columnas; i++) {
        matriz[i] = (double)rand() / RAND_MAX;
    }
}

void mostrar_matriz(double *matriz, int filas, int columnas) {
    for (int f = 0; f < filas; f++) {
        for (int c = 0; c < columnas; c++) {
            printf("%lf ", matriz[f * columnas + c]);
        }
        printf("\n");
    }
}

void producto_matvec(double *local_A, double *local_x, double *local_y,
                     int filas_locales, int n, int columnas_locales, MPI_Comm comunicador) {

    double *x_total = (double *)malloc(n * sizeof(double));

    MPI_Allgather(local_x, columnas_locales, MPI_DOUBLE, x_total, columnas_locales, MPI_DOUBLE, comunicador);

    for (int i = 0; i < filas_locales; i++) {
        double suma = 0.0;
        for (int j = 0; j < n; j++) {
            suma += local_A[i * n + j] * x_total[j];
        }
        local_y[i] = suma;
    }

    free(x_total);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rango, tamanio;
    int m = 9, n = 9;
    MPI_Comm_rank(MPI_COMM_WORLD, &rango);
    MPI_Comm_size(MPI_COMM_WORLD, &tamanio);

    int filas_locales = m / tamanio;
    int columnas_locales = n / tamanio;

    double *local_A = (double *)calloc(filas_locales * n, sizeof(double));
    double *local_x = (double *)calloc(columnas_locales, sizeof(double));
    double *local_y = (double *)calloc(filas_locales, sizeof(double));

    llenar_matriz(local_A, filas_locales, n);
    llenar_matriz(local_x, columnas_locales, 1);

    producto_matvec(local_A, local_x, local_y, filas_locales, n, columnas_locales, MPI_COMM_WORLD);

    if (rango == 0) {
        double *y = (double *)malloc(m * sizeof(double));
        MPI_Gather(local_y, filas_locales, MPI_DOUBLE, y, filas_locales, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        printf("Vector resultado:\n");
        for (int i = 0; i < m; i++) {
            printf("%lf ", y[i]);
        }
        printf("\n");
        free(y);
    } else {
        MPI_Gather(local_y, filas_locales, MPI_DOUBLE, NULL, filas_locales, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    free(local_A);
    free(local_x);
    free(local_y);

    MPI_Finalize();
    return 0;
}
