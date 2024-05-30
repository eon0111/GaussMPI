/**
 * @file gauss.c
 * @author Noé Ruano Gutiérrez (nrg916@alumnos.unican.es)
 * @brief Implementación distribuida y paralelizada del algoritmo de Gauss-Jordan para la resolución
 * de sistemas de ecucaciones lineales.
 * @version 0.1
 * 
 */

/***************************************************************************************************
 * NOTE: esta versión de la implementación pretendía realizar un reparto de la matriz ampliada por *
 * columnas, haciendo que cada thread con su chunk de columnas procediese del modo siguiente:      *
 * 1. En primer lugar, debería actualizar los coeficientes por encima del pivote de la columna     *
 *    conforme a las transformaciones realizadas teniendo en cuenta los pivotes de las columnas    *
 *    anteriores                                                                                   *
 * 2. En segundo lugar, transformaría en cero los coeficientes por debajo del pivote, conforme a   *
 *    los ratios calculados en base al valor del pivote de la columna, y el valor de estos pivotes *
 * No obstante, a pesar de parecer a priori una solución que podría llevar a buen puerto, no es    *
 * viable por un pequeño detalle, y es que para realizar la actualización de los coeficientes por  *
 * encima del pivote no basta con tener todos los pivotes de la matriz que representa el sistema   *
 * en su estado inicial sino que, conforme avanza el algoritmo, el valor de los pivotes también    *
 * cambia, puesto que son afectados por las transformaciones de filas que se van haciendo. Es por  *
 * esto que la implementación se tornaría secuencial, al no poder concluir la actualización de los *
 * coefs. por encima del pivote hasta haberse actualizado todos los pivotes de las columnas        *
 * anteriores.                                                                                     *
 **************************************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>
#include "mpi.h"

#define RANGO_ROOT	0	// El rango del nodo raíz

/**
 * @brief Computa el número de elementos situados bajo la diagonal de la matriz en base a la
 * dimensión de esta.
 */
#define n_elems_triang(dim) (0.5 * dim * (dim - 1))

/**
 * @brief Retorna el número de columnas que recibirá un proceso en función de si su rango corresponde
 * con un proceso de los nProcs - 1 primeros procesos en el comunicador, o si es el último proceso.
 */
typedef enum {
	NORMAL, ULTIMO
} tipo_proceso;
#define n_cols(tipo, dim) ((tipo == NORMAL) ? (int)ceil((float)(dim + 1) / (float)n_procs) : (int)((dim + 1) - (ceil((float)(dim + 1) / (float)n_procs)) * (n_procs - 1)))

/**
 * @brief Muestra una matriz en la salida estándar.
 * @param matriz la matriz
 * @param dim_x el número de columnas de la matriz
 * @param dim_y el número de filas de la matriz
 */
void muestra_matriz(float *matriz, int dim_x, int dim_y) {
	printf("[*] MATRIZ (%dx%d):\n", dim_x, dim_y);

	int i, j;
	for (j = 0; j < dim_y; j++)
	{
		for (i = 0; i < dim_x; i++)
			printf("%6.2f", matriz[j * dim_x + i]);
		printf("\n");
	}
}

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
 * @brief Diagonaliza la matriz ampliada que representa el sistema a resolver.
 * 	
 * @param A puntero al array que alberga las columnas de la matriz ampliada sobre las que deba
 * trabajar el thread
 * @param p puntero al array de pivotes
 * @param n el número de columnas que conforman la carga de trabajo del thread
 */
void escalona_matriz(float *A, const float *p, int n)
{
	int i, j, count;
	float ratio;

	/* Eliminación Gaussiana */
	for (i = 0; i < (n - 1); i++)
	{
		/* -- 3 -- Actualización de los eltos. de la columna ------------------------------------ */
		/* Antes de proceder con el escalonado de la matriz, el thread debe realizar sobre cada elto.
		 * de cada una de las columnas que conforman su carga de trabajo, todas las operaciones que
		 * corresponden a las sucesivas acutalizaciones de esos eltos. realizadas a lo largo de la
		 * ejecución del algoritmo y que, en la versión secuencial del mismo, corresponden con
		 * operaciones a nivel de fila. */

		for (int k = 1; k < n; k++)
		{
			ratio = A[k * n + i] / p[k];
			A[k * n + i] -= (ratio * A[i * n + count]);
		}

		/* Hace ceros por debajo del pivote */
		for (j = (i + 1); j < n; j++)
		{
			/* ratio es el cociente del elto. de la columna que se está observando entre el pivote de esa columna */
			ratio = A[j * n + i] / A[i * n + i];	// i: columna --- j: fila
														// A[j * n][i] -> eltos. de cada columna
														// A[i * n][i] -> pivotes (eltos. en la diagonal)
			/* Ahora se computa la operación con la fila completa, haciendo un cero debajo del pivote
			 * y alterando el resto de eltos. de la fila conforme al valor del pivote. En la siguiente
			 * iteración (j) se hace otro cero por debjo del pivote y se computa el resto de la fila
			 * que se esté observando, etc. */
			for (count = i; count < n; count++)
				A[j * n + count] -= (ratio * A[i * n + count]);
		}
	}
}

void resuelve_sistema(const float *A, const float *b, float *x, int n)
{
	float *Acpy = (float *) calloc(n * n, sizeof(float));
	float *bcpy = (float *) malloc(n * sizeof(float));
	memcpy(Acpy, A, n * n * sizeof(float));
	memcpy(bcpy, b, n * sizeof(float));

	int i, j;
	
	/* Back-substitution */
	x[n - 1] = bcpy[n - 1] / Acpy[(n - 1) * n + n - 1];
	for (i = (n - 2); i >= 0; i--) {
		float temp = bcpy[i];
		for (j = (i + 1); j < n; j++) {
			temp -= (Acpy[i * n + j] * x[j]);
		}
		x[i] = temp / Acpy[i * n + i];
	}
}

int main(int argc, char **argv)
{
	int dim = 5, mi_rango;
	char *filename = "results.out";
	
	if (argc > 1)
		dim = atoi(argv[1]);

	float *A = (float *)malloc((dim + 1) * dim * sizeof(float)); // Matriz de coeficientes
	float *x = (float *)malloc(dim * sizeof(float));			 // Vector de resultados

	genera_sistema_lineal(dim, A);

	/* Comienzo de la región distribuida */
	MPI_Init(NULL, NULL);

	int n_procs;
	MPI_Status status;
	MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &mi_rango);

#ifdef DEBUG
	printf("[*] Soy el worker %d\n", mi_rango);
#endif

	/* -- 1 -- Distribución de los pivotes ------------------------------------------------------ */
	/* Cada worker necesita contar con todos los pivotes de la matriz para poder realizar todas las
	 * actualizaciones de los eltos. de la matriz situados por encima del pivote de cada columna sobre
	 * la que vaya a trabajar. */

	if (mi_rango == 0)
	{
		/* Construcción de un nuevo tipo de dato para distribuir los pivotes (eltos. de la diagonal
		 * principal de la matriz) */
		MPI_Datatype tipo_pivotes;
		MPI_Type_vector(dim, 1, dim + 2, MPI_FLOAT, &tipo_pivotes);
		MPI_Type_commit(&tipo_pivotes);
		
		/* Envío de los pivotes al resto de workers y liberación del tipo de dato */
		MPI_Bcast(A, 1, tipo_pivotes, mi_rango, MPI_COMM_WORLD);
		MPI_Type_free(&tipo_pivotes);

		/* -- 2 -- Distribución de las columnas de la matriz ampliada --------------------------- */
		/* - En el reparto de las cargas de trabajo se sigue una metodología "round robin", es decir,
		 * que se van asignando columnas de la matriz ampliada a los threads siguiendo esa política,
		 * de forma que el reparto de las cargas sea lo más equitativo posible, puesto que la carga
		 * que supondrá el procesamiento de una columna de la matriz dependerá del número de eltos.
		 * emplazados en filas cuyos índices las sitúen por encima del elto. pivote de esa columna,
		 * al tener que actualizar el thread todos esos eltos. conforme a los pivotes de las columnas
		 * anteriores, algunas de las cuales ya habrá procesado.
		 * 
		 * - Cada thread puede inferir el pivote a emplear en el procesamiento de cada elto. de cada
		 * columna en base a su rango, el número de threads en el comunicador, la dimensión de la
		 * matriz y la posición que ocupe la matriz dentro del array donde se almacenen las columnas
		 * a procesar. */

		/* NOTE: en verdad, puedo repartir la matriz en bloques de columnas, porque el proceso de
		 * actualización de los eltos. por encima de la diagonal supone la misma carga de cómputo
		 * que la obtención de ceros por debajo de la diagonal. En consecuencia, el primer proceso
		 * realizará pocas actualizaciones y muchos ceros y, el último, pocos ceros y muchas
		 * actualizaciones, con lo que las cargas de trabajo de los procesos MPI quedarán igualadas.

		/* Construcción de un nuevo tipo de dato para distribuir las columnas de la matriz */
		MPI_Datatype tipo_columnas;

		/* Redondea el número de columnas hacia arriba para que el último thread no se cargue con
		 * todo el trabajo restante del reparto, sino que dicho trabajo se reparta entre los threads
		 * anteriores */
		MPI_Type_vector(dim, n_cols(NORMAL,dim), dim + 1, MPI_FLOAT, &tipo_columnas);
		MPI_Type_commit(&tipo_columnas);

		/* Envío de las columnas a los primeros (n_procs - 1) procesos */
		for (int i = 1; i < n_procs - 1; i++)
			MPI_Send(&A[i * n_cols(NORMAL,dim)], 1, tipo_columnas, i, 0, MPI_COMM_WORLD);
		
		/* Dado que la carga de trabajo del thread con mayor rango difiere de la del resto de threads
		 * que le preceden, es preciso ajustar el tipo de dato a enviar para contemplar su caso */
		MPI_Type_free(&tipo_columnas);
		MPI_Type_vector(dim, n_cols(ULTIMO,dim), dim + 1, MPI_FLOAT, &tipo_columnas);
		MPI_Type_commit(&tipo_columnas);

		MPI_Send(&A[n_procs - 1], 1, tipo_columnas, n_procs - 1, 0, MPI_COMM_WORLD);

		MPI_Type_free(&tipo_columnas);
	}
	else
	{
		/* Recepción de los pivotes */
		float *pivotes = (float *)malloc(dim * sizeof(float));
		MPI_Bcast(pivotes, dim, MPI_FLOAT, 0, MPI_COMM_WORLD);

#ifdef DEBUG
		printf("[*] Soy [%d]. Pivotes recibidos: ", mi_rango);
		for (int i = 0; i < dim; i++)
			printf(" [%5.2f]", pivotes[i]);
		printf("\n");
#endif

		/* Recepción de las columnas */
		float *columnas;
		if (mi_rango != n_procs - 1)
		{
			columnas = (float *)malloc(sizeof(float) * dim * n_cols(NORMAL,dim));
			MPI_Recv(columnas, dim * n_cols(NORMAL,dim), MPI_FLOAT, RANGO_ROOT, 0, MPI_COMM_WORLD, &status);
		}
		else
		{
			columnas = (float *)malloc(sizeof(float) * dim * n_cols(ULTIMO,dim));
			MPI_Recv(columnas, dim * n_cols(ULTIMO,dim), MPI_FLOAT, RANGO_ROOT, 0, MPI_COMM_WORLD, &status);
		}

		free(pivotes);
	}

	// solve_linear_system(A, b, x, dim);

	/* Fin de la región distribuida */
	MPI_Finalize();

	// escribe_resultados (x, dim, filename);

	// printf("Done! Results in the file: %s\n", filename);

	return EXIT_SUCCESS;
}
