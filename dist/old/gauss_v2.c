/**
 * @file gauss.c
 * @author Noé Ruano Gutiérrez (nrg916@alumnos.unican.es)
 * @brief Implementación distribuida y paralelizada del algoritmo de Gauss-Jordan para la resolución
 * de sistemas de ecuaciones lineales.
 * @version 0.2
 * 
 */

/***************************************************************************************************
 * NOTE: esta es una versión preliminar de gauss_v3                                                *
 **************************************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>
#include "mpi.h"

#define ROOT 0	// El rango del proceso root

/**
 * @brief Computa el número de elementos situados bajo la diagonal de la matriz en base a la
 * dimensión de esta.
 */
#define n_elems_triang(dim) (0.5 * dim * (dim - 1))

/**
 * @brief Retorna el número de columnas que recibirá un proceso en función de si su rango corresponde
 * con un proceso de los nProcs - 1 primeros procesos en el comunicador, o si es el último proceso.
 */
#define n_elems(n, num_threads, tipo) ((tipo == NORMAL) ? (n) / num_threads : (n) - (num_threads - 1) * ((n) / num_threads))

/**
 * @brief Retorna el número de columnas que recibirá un proceso en función de si su rango corresponde
 * con un proceso de los nProcs - 1 primeros procesos en el comunicador, o si es el último proceso.
 */
typedef enum {
	NORMAL, ULTIMO
} tipo_proceso;
#define n_filas(n, id, procs) ()

/**
 * @brief Escribe los resultados de la resolución en un fichero.
 * 
 * @param x el vector con las soluciones al sistema
 * @param n la longitud del vector de soluciones, es decir, la dimensión de la matriz de coeficientes
 * @param filename el nombre del fichero de salida
 * @return 0 si se produce una terminación correcta de la función, 1 en caso contrario (no puede
 * abrirse el fichero)
 */
int escribe_resultados(float *x, int n, char *filename)
{
	int i;
	FILE *fp;

	if ((fp = fopen(filename, "w")) == NULL)
	{
		printf("Error: No puedo abrir el fichero %s\n", filename);
		exit(1);
	}

	for (i = 0; i < n; i++)
		fprintf(fp, "x[%d] = %.4f\n", i, x[i]);

	fclose(fp);
	return 0;
}

/**
 * @brief Genera la matriz ampliada que representa el sistema lineal a resolver.
 * 
 * @param n la dimensión de la matriz de coeficientes
 * @param A puntero al vector que alberga la matriz ampliada
 */
// TODO: susceptible de ser paralelizado a nivel local
void genera_sistema_lineal(int dim, float *A) 
{
	int i, j;
	for (i = 0; i < dim; i++) {
		for (j = 0; j < dim; j++)
			A[i * (dim + 1) + j] = (1.0 * dim + (rand() % dim)) / (i + j + 1);
		/* Eltos. en la diagonal */
		A[i * (dim + 1) + i] = (10.0 * dim) / (i + i + 1);
		/* Última columna de la matriz ampliada */
		A[i * (dim + 1) + dim] = 1.0;
	}
}

/**
 * @brief Muestra una matriz en la salida estándar.
 * @param matriz la matriz
 * @param dim_x el número de columnas de la matriz
 * @param dim_y el número de filas de la matriz
 */
void muestra_matriz(float *matriz, int dim_x, int dim_y)
{
	printf("[*] MATRIZ (%dx%d):\n", dim_x, dim_y);

	int i, j;
	for (j = 0; j < dim_y; j++)
	{
		for (i = 0; i < dim_x; i++)
			printf("%6.2f", matriz[j * dim_x + i]);
		printf("\n");
	}
}

int main(int argc, char **argv)
{

	int dim = 4;
	char *filename = "results.out";
	
	if (argc > 1) dim = atoi(argv[1]);

	float *A = (float *)malloc((dim + 1) * dim * sizeof(float)); // Matriz de coeficientes
	float *x = (float *)malloc(dim * sizeof(float));			 // Vector de resultados

	/* Construcción del sistema lineal de ecuaciones (matriz ampliada A*)*/
	genera_sistema_lineal(dim, A);

	int i, j, count, mi_rango, n_procs;
	float ratio;

	/* Comienzo de la sección de comunicaciones MPI */
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &mi_rango);
	MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

	/* Vectores en los que se guarda el número de eltos. que recibirá cada thread, y los desplazamientos
	 * desde el comienzo del buffer de envío de MPI_Scatterv hasta el primer elto. de la sección del
	 * thread */
	int *counts = (int *)malloc(sizeof(int) * n_procs);
	int *displs = (int *)malloc(sizeof(int) * n_procs);

	for (i = 0; i < n_procs; i++)
	{
		counts[i] = n_elems(dim + 1, n_procs, ((i == n_procs - 1) ? ULTIMO : NORMAL));
		displs[i] = i * n_elems(dim + 1, n_procs, NORMAL);
	}

	/* El no. de eltos. que constituye la sección de la fila sobre la que trabajará cada thread */
	int n_elems_segmento = n_elems(dim + 1, n_procs, ((mi_rango == n_procs - 1) ? ULTIMO : NORMAL));
	/* La fila a la que pertenece el pivote */
	float *fila_referencia = (float *)malloc(sizeof(float) * n_elems_segmento);
	/* La fila con la que va a realizarse la operación */
	float *fila_resultados = (float *)malloc(sizeof(float) * n_elems_segmento);

	if (mi_rango == 0)
	{
		muestra_matriz(A, dim + 1, dim);
		
		/* Eliminación Gaussiana */
		for (i = 0; i < dim; i++) {
				/* Scatter de la fila de referencia (aquella a la que pertenece el pivote) */
				MPI_Scatterv(&A[i * (dim + 1)], counts, displs, MPI_INT, fila_referencia, n_elems_segmento, MPI_INT, 0, MPI_COMM_WORLD);
#ifdef DEBUG
				printf("[0] Fila [%d] enviada (referencia)\n", i);
#endif
			for (j = (i + 1); j < dim + 1; j++) {
#ifdef DEBUG
				printf("[0] Ratio [%d,%d] ----- dim -> %d ----- A[j*(dim+1)+i] = %.3f ----- Pivote [%d,%d] = %.3f\n", i, j, dim, A[j * (dim + 1) + i], i, j, A[i * (dim + 1) + i]);
#endif
				/* "ratio" es el cociente del elto. de la fila 'j' entre el pivote de la columna 'i' */
				ratio = A[j * (dim + 1) + i] / A[i * (dim + 1) + i];	// i: columna --- j: fila
																		// A[j * (dim + 1)][i] -> eltos. de cada columna
																		// A[i * (dim + 1)][i] -> pivotes (eltos. en la diagonal)

				/* Envío del ratio para que no tengan que calcularlo todos los threads */
				MPI_Bcast(&ratio, 1, MPI_FLOAT, mi_rango, MPI_COMM_WORLD);
#ifdef DEBUG
				printf("[0] Ratio [%d,%d] enviado -> %.3f\n", i, j, ratio);
#endif

				/* Scatter de la fila con la que se opera y sobre la que se almacenan los resultados
				 * de la transformación conforme al ratio */
				MPI_Scatterv(&A[j * (dim + 1)], counts, displs, MPI_INT, fila_resultados, n_elems_segmento, MPI_INT, 0, MPI_COMM_WORLD);
#ifdef DEBUG
				printf("[0] Fila  [%d] enviada (resultados)\n", j);
#endif

				/* NOTE: Ahora se computa la operación con la fila completa, haciendo un cero debajo del
				 * pivote y alterando el resto de eltos. de la fila conforme al valor del pivote. En
				 * la siguiente iteración (j) se hace otro cero por debjo del pivote y se computa el
				 * resto de la fila que se esté observando, etc. */
				for (count = i; count < n_elems_segmento; count++)	// Llega a dim + 1 porque trabaja con A7*
					fila_resultados[count] -= (ratio * fila_referencia[count]);

				/* Recuperación de los resultados obtenidos por el resto de threads */
				MPI_Gatherv(fila_resultados, n_elems_segmento, MPI_INT, &A[j * (dim + 1)], counts, displs, MPI_INT, 0, MPI_COMM_WORLD);
#ifdef DEBUG
				printf("[0] Fila %d recuperada\n", j);
#endif
			}

			muestra_matriz(A, dim + 1, dim);
		}
	}
	else
	{
		for (i = 0; i < dim; i++) {
			/* Recepción de la fila de referencia */
			MPI_Scatterv(&A[i * (dim + 1)], counts, displs, MPI_INT, fila_referencia, n_elems_segmento, MPI_INT, 0, MPI_COMM_WORLD);
#ifdef DEBUG
			printf("[%d] Fila [%d] recibida (referencia)\n", mi_rango, i);
#endif
			for (j = (i + 1); j < dim + 1; j++) {
				/* Recepción del ratio */
				MPI_Bcast(&ratio, 1, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
#ifdef DEBUG
				printf("[%d] Ratio recibido -> %.3f\n", mi_rango, ratio);
#endif

				/* Recepción de la fila de resultados */
				MPI_Scatterv(&A[j * (dim + 1)], counts, displs, MPI_INT, fila_resultados, n_elems_segmento, MPI_INT, 0, MPI_COMM_WORLD);
#ifdef DEBUG
				printf("[%d] Fila [%d] recibida (resultados)\n", mi_rango, j);
#endif

				/* Realiza la transformación de la fila de resultados conforme al ratio */
				for (count = i; count < n_elems_segmento; count++)	// Llega a dim + 1 porque trabaja con A7*
					fila_resultados[count] -= (ratio * fila_referencia[count]);

				/* Envío de los resultados al thread root */
				/* Recuperación de los resultados obtenidos por el resto de threads */
				MPI_Gatherv(fila_resultados, n_elems_segmento, MPI_INT, &A[j * (dim + 1)], counts, displs, MPI_INT, 0, MPI_COMM_WORLD);
			}
		}
	}

	MPI_Finalize();

	/* Back-substitution */
	// x[n - 1] = bcpy[n - 1] / Acpy[(n - 1) * n + n - 1];
	// for (i = (n - 2); i >= 0; i--) {
	// 	float temp = bcpy[i];
	// 	for (j = (i + 1); j < n; j++) {
	// 		temp -= (Acpy[i * n + j] * x[j]);
	// 	}
	// 	x[i] = temp / Acpy[i * n + i];
	// }

	// resuelve_sistema_lineal(A, b, x, dim);

	// escribe_resultados (x, dim, filename);

	// printf("Done! Results in the file: %s\n", filename);

	return EXIT_SUCCESS;
}

