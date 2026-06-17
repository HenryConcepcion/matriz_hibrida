#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#define N 4

int main(int argc, char *argv[]) {

    int rank, size;
    int A[N][N], B[N][N], C[N][N];

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int filas_por_proceso = N / size;

    int subA[filas_por_proceso][N];
    int subC[filas_por_proceso][N];

    if(rank == 0) {

        for(int i=0;i<N;i++) {
            for(int j=0;j<N;j++) {

                A[i][j] = i + j;
                B[i][j] = i * j;
            }
        }
    }

    double inicio = MPI_Wtime();

    MPI_Bcast(B, N*N, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Scatter(
        A,
        filas_por_proceso * N,
        MPI_INT,
        subA,
        filas_por_proceso * N,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < filas_por_proceso; i++) {

        for(int j = 0; j < N; j++) {

            subC[i][j] = 0;

            for(int k = 0; k < N; k++) {

                subC[i][j] += subA[i][k] * B[k][j];
            }
        }
    }

    MPI_Gather(
        subC,
        filas_por_proceso * N,
        MPI_INT,
        C,
        filas_por_proceso * N,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    double fin = MPI_Wtime();

    if(rank == 0) {

        printf("\nMatriz resultado:\n\n");

        for(int i=0;i<N;i++) {

            for(int j=0;j<N;j++) {

                printf("%4d ", C[i][j]);
            }

            printf("\n");
        }

        printf("\nTiempo de ejecucion: %f segundos\n",
               fin - inicio);
    }

    MPI_Finalize();

    return 0;
}
