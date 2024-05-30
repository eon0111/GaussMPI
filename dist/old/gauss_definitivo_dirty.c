/**
 * @file gauss.c
 * @author Noé Ruano Gutiérrez (nrg916@alumnos.unican.es)
 * @brief Implementación distribuida y paralelizada del algoritmo de Gauss-Jordan para la resolución
 * de sistemas de ecuaciones lineales.
 * @version 0.4
 * 
 */

/***************************************************************************************************
 * NOTE: esta es la versión empleada en la depuración de la versión final                          *
 **************************************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>
#include <stdbool.h>
#include "mpi.h"

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
		printf("[!] ERROR: No pudo abrirse el fichero \"%s\"\n", filename);
		exit(EXIT_FAILURE);
	}

	for (i = 0; i < n; i++)
		fprintf(fp, "x[%d] = %f\n", i, x[i]);

	fclose(fp);
	return 0;
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

		// Eltos. en la diagonal
		A[i * (dim + 1) + i] = (10.0 * dim) / (i + i + 1);

		// Última columna de la matriz ampliada
		A[i * (dim + 1) + dim] = 1.0;
	}
}

/**
 * @brief Inicializa el vector de counts para usar en el scatter.
 * 
 * @param counts puntero al vector
 * @param num_procs el número de procesos MPI que trabajarán sobre las num_filas filas que
 * correspondan
 * @param num_eltos el número de eltos. a repartir entre los num_procs procesos MPI
 * @param dim el no. de eltos. de la fila o columna a repartir
 * @param filas indica si el reparto será de eltos. en una columna (false) o en una fila (true)
 */
void inicializa_counts(int *counts, int num_procs, int num_eltos, int dim, bool filas)
{
	// El no. de filas sobre las que operará(n) el/los primer(os) proceso(s)
	int count_a = ceil((float)num_eltos / num_procs);

	// El no. de filas sobre las que operará(n) el/los restante(s)
	int count_b = count_a - 1;

	int mod = num_eltos % num_procs;

	int i;
	for (i = 0; i < ((mod == 0) ? num_procs : mod); i++)
		counts[i] = count_a * (((num_eltos <= num_procs) && filas) ? 1 : dim); // TODO: hacer un apaño menos guarro
	for (; i < num_procs; i++)
		counts[i] = count_b * dim;
}

/**
 * @brief Inicializa el vector de desplazamientos para usar en el scatter.
 * 
 * @param displs puntero al vector de desplazamientos
 * @param num_procs el número de procesos MPI que trabajarán sobre las filas que corresponda
 * @param counts puntero al vector de counts, empleado en el cálculo de los desplazamientos
 */
void inicializa_displs(int *displs, int num_procs, int *counts)
{
	int i, j;
	for (i = 0; i < num_procs; i++)
	{
		displs[i] = 0;
		for (j = 0; j < i; j++)
			displs[i] += counts[j];
	}
}

void main(int argc, char **argv)
{

	int dim = 4;
	char *filename = "results.out";
	
	if (argc > 1) dim = atoi(argv[1]);

	float *A = (float *)malloc((dim + 1) * dim * sizeof(float)); // Matriz de coeficientes
	float *x = (float *)calloc(dim, sizeof(float));			 // Vector de resultados

	int i, j, k, count, mi_rango, num_procs;
	float ratio, t_inicio, t_fin, t_total;

	// Comienzo de la sección de comunicaciones MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &mi_rango);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	MPI_Barrier(MPI_COMM_WORLD);
	if (mi_rango == 0) t_inicio = MPI_Wtime();

	/* Vectores en los que se guarda el número de eltos. que recibirá cada thread, y los desplazamientos
	 * desde el comienzo del buffer de envío de MPI_Scatterv hasta el primer elto. del bloque sobre
	 * el que trabajará el thread */
	int *counts = (int *)malloc(sizeof(int) * num_procs);
	int *displs = (int *)calloc(num_procs, sizeof(int));

	// La fila a la que pertenece el pivote
	float *fila_referencia = (float *)malloc(sizeof(float) * (dim + 1));
	
	// El bloque de filas con el que van a realizarse las operaciones
	inicializa_counts(counts, num_procs, dim - 1, dim + 1, false);
	inicializa_displs(displs, num_procs, counts);
	float *bloque_resultados = (float *)malloc(sizeof(float) * counts[mi_rango]);

	// (Sólo el ROOT) -> construcción del sistema lineal de ecuaciones (matriz ampliada A*)
	if (mi_rango == 0) genera_sistema_lineal(dim, A);

	/*************************
	 * ELIMINACIÓN GAUSSIANA *
	 ************************/
	for (i = 0; i < dim - 1; i++)
	{
#ifdef DEBUG
		if (mi_rango == 0) muestra_matriz(A, dim + 1, dim);

		// printf("[%d] counts ->", mi_rango);
		// for (int k = 0; k < num_procs; k++)
		// 	printf(" [%d]", counts[k]);
		// printf("\n");

		// printf("[%d] displs ->", mi_rango);
		// for (int k = 0; k < num_procs; k++)
		// 	printf(" [%d]", displs[k]);
		// printf("\n");
#endif
		// Broadcast de la fila de referencia (aquella a la que pertenece el pivote)
		MPI_Bcast(((mi_rango == 0) ? &A[i * (dim + 1)] : fila_referencia), dim + 1, MPI_FLOAT, 0,
				  MPI_COMM_WORLD);
#ifdef DEBUG
		// printf("[%d] Fila referencia recibida ->", mi_rango);
		// for (int k = 0; k < dim + 1; k++)
		// 	printf(" [%.2f]", ((mi_rango == 0) ? A[i * (dim + 1) + k] : fila_referencia[k]));
		// printf("\n");
#endif
		/*******************************************************************************************
		 * Scatter de todas las filas por debajo de la de referencia. Se hace scatter siempre a    *
		 * partir del primer elto. de cada fila, es decir, que los threads reciben bloques         *
		 * completos de la matriz, aunque reciban columnas que ya han sido reducidas (columnas con *
		 * ceros). Esto se hace así para evitar tener que construir un array de desplazamientos    *
		 * descomunal que habría que actualizar en cada iteración (i).                             *
		 * En resumen, que por cada iteración i no se mandan solo bloques de la matriz a partir de *
		 * la columna i, sino que se envían todos los eltos. antes y después de la columna i       *
		 ******************************************************************************************/
#ifdef DEBUG
		// printf("[%d] Recibiendo %d eltos. a partir de A[%d] con despl. de %d eltos.\n", mi_rango, counts[mi_rango], (i + 1) * (dim + 1), displs[mi_rango]);
#endif
		MPI_Scatterv(&A[(i + 1) * (dim + 1)], counts, displs, MPI_FLOAT, bloque_resultados,
					 counts[mi_rango], MPI_FLOAT, 0, MPI_COMM_WORLD);
#ifdef DEBUG
		// printf("[%d] Bloque de resultados recibido:", mi_rango);
		// for (int k = 0; k < counts[mi_rango] / (dim + 1); k++)
		// 	for (int a = 0; a < dim + 1; a++)
		// 		printf(" [%.2f]", bloque_resultados[k * (dim + 1) + a]);
		// printf("\n");
#endif
		// Reduce por debajo del pivote y actualiza las filas conforme al ratio asociado a cada una
		for (j = 0; j < counts[mi_rango] / (dim + 1); j++)
		{
			/***************************************************************************************
			 * "ratio" es el cociente del elto. de la fila 'j' entre el pivote de la columna 'i'   *
			 *                                                                                     *
			 * A[j * (dim + 1)][i]	-> eltos. de cada columna                                      *
			 * fila_referencia[i]	-> pivotes (eltos. en la diagonal)                             *
			 **************************************************************************************/
			ratio = bloque_resultados[j * (dim + 1) + i] / ((mi_rango == 0) ? A[i * (dim + 1) + i] : fila_referencia[i]);
#ifdef DEBUG
			// if (mi_rango == 0) printf("[%d] ratio (j = %d --- %.2f / %.2f = %.7f\n", mi_rango, j, bloque_resultados[j * (dim + 1) + i], ((mi_rango == 0) ? A[i * (dim + 1) + i] : fila_referencia[i]), ratio);
#endif
			/***************************************************************************************
			 * Ahora se computa la operación con la fila completa, haciendo un cero debajo del     *
			 * pivote y alterando el resto de eltos. de la fila conforme al valor del pivote. En   *
			 * la siguiente iteración (j) se hace otro cero por debajo del pivote y se computa el  *
			 * resto de la fila que se esté observando, etc.                                       *
			 * Nótese que se computan los eltos. a partir de aquel en la columna de índice 'i'     *
			 * (alineado con el pivote), puesto que los eltos. anteriores en la fila ya habŕan     *
			 * sido reducidos (transformados en 0) en iteraciones (i) anteriores                   *
			 **************************************************************************************/
			for (count = i; count < dim + 1; count++)
				bloque_resultados[j * (dim + 1) + count] -= (ratio * ((mi_rango == 0) ? A[i * (dim + 1) + count] : fila_referencia[count]));
#ifdef DEBUG
			// printf("[%d] Bloque procesado:", mi_rango);
			// for (int k = 0; k < counts[mi_rango] / (dim + 1); k++)
			// 	for (int a = 0; a < dim + 1; a++)
			// 		printf(" [%.2f]", bloque_resultados[k * (dim + 1) + a]);
			// printf("\n");
#endif
		}

		// Unificación de los resultados obtenidos por todos los threads sobre la matriz del root
		MPI_Gatherv(bloque_resultados, counts[mi_rango], MPI_FLOAT, &A[(i + 1) * (dim + 1)], counts,
					displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

		/* Ajusta los vectores de no. de eltos. por thread y desplazamientos de forma que se realice
		 * un reparto equitativo de las filas situadas por debajo de la fila del pivote */
		inicializa_counts(counts, num_procs, dim - i - 2, dim + 1, false);
		inicializa_displs(displs, num_procs, counts);
	}
#ifdef DEBUG
		if (mi_rango == 0) muestra_matriz(A, dim + 1, dim);
#endif

	/***************************/
	/* Sustitución hacia atrás */
	/***********************************************************************************************
	 * En cada fila, para computar el valor de la incógnita que acompaña al pivote de esa fila, se *
	 * resta al término independiente (el último elto. de la fila, perteneciente a la columna con  *
	 * la que se construye A*), todos los productos de los coeficientes en su misma fila (a partir *
	 * del pivote, no incluído), por el valor de las incógnitas despejadas en iteraciones (i)      *
	 * anteriores.                                                                                 *
	 * Al final, una vez terminado este cómputo, se realiza el despeje de la incógnita que         *
	 * acompaña al pivote -> a * x = b -> x = b / a, siendo:                                       *
	 *    - x : x[i], los valores de las incógnitas despejadas con anterioridad a la               *
	 *          actual.                                                                            *
	 *    - b : temp, la variable donde se acumulan las sustracciones.                             *
	 *    - a : A[i * (dim + 1) + i], el coeficiente que acompaña a x[i], es decir, el             *
	 *          pivote de la fila.                                                                 *
	 ***********************************************************************************************
	 *                                                                                             *
	 *    1.- Reparto de las incógnitas despejadas hasta el momento                                *
	 *    2.- Reparto de cada fila desde i hasta dim - 2                                           *
	 *    3.- Cada uno computa sustracciones por su cuenta, pero no sobre el término independiente *
	 *    4.- Recolección de los resultados de los threads, un valor cada uno                      *
	 *    5.- El root resta al término independiente los valores calculados por cada thread y hace *
	 *        el despeje                                                                           *
	 *                                                                                             *
	 **********************************************************************************************/
	
	// Se computa el primer despeje, que es directo
	if (mi_rango == 0) x[dim - 1] = A[(dim + 1) * dim - 1] / A[(dim + 1) * dim - 2];

	/* Actualiza los counts y displs para usarlos más adelante en el scatter y repartir los
	 * eltos. de las filas de manera equitativa */
	inicializa_counts(counts, num_procs, 1, 1, true);
	inicializa_displs(displs, num_procs, counts);

	/* - seccion_fila : vectores donde los threads reciben secciones de cada fila
	 * - seccion_incognitas : vector donde los threads reciben secciones del vector de resultados
	 * - resultados : vector donde el root guarda los resultados individuales calculados por el
	 *                resto de threads */
	float *seccion_fila = (float *)malloc(sizeof(float) * dim / num_procs);
	float *seccion_incognitas = (float *)malloc(sizeof(float) * dim / num_procs);
	float *resultados = (float *)calloc(num_procs, sizeof(float));

	// Variable donde cada thread almacena su resultado individual 
	float mi_resultado;

	// Variable donde el root acumula los resultados individuales de todos los threads
	float temp;

	// Se computa el valor del resto de incógnitas
	for (i = (dim - 2); i >= 0; i--)
	{
#ifdef DEBUG
		// if (mi_rango == 0)
		// {
		// 	printf("[*] counts ->");
		// 	for (k = 0; k < num_procs; k++)
		// 		printf(" [%d]", counts[k]);
		// 	printf("\n");
		
		// printf("[*] displs ->");
		// for (k = 0; k < num_procs; k++)
		// 	printf(" [%d]", displs[k]);
		// printf("\n");
		// }
#endif
		// Scatter del vector de resultados (incógnitas despejadas)
		MPI_Scatterv(&x[i + 1], counts, displs, MPI_FLOAT, seccion_incognitas, counts[mi_rango], MPI_FLOAT, 0, MPI_COMM_WORLD);
#ifdef DEBUG
		printf("[%d] incógnitas ->", mi_rango);
		for (k = 0; k < counts[mi_rango]; k++)
			printf(" [%f]", seccion_incognitas[k]);
		printf("\n");
#endif

		// Scatter de la fila i a partir del pivote (no incluido)
		MPI_Scatterv(&A[i * (dim + 1) + i + 1], counts, displs, MPI_FLOAT, seccion_fila, counts[mi_rango], MPI_FLOAT, 0, MPI_COMM_WORLD);
#ifdef DEBUG
		// printf("[%d] Sección de fila recibida ->", mi_rango);
		// for (k = 0; k < counts[mi_rango]; k++)
		// 	printf(" [%.2f]", seccion_fila[k]);
		// printf("\n");
#endif

		/* El root inicializa la variable acumuladora a la que restará más adelante los valores
		 * calculados por cada thread */
		if (mi_rango == 0) temp = A[(dim + 1) * i + dim];

		// Cada thread computa las sustracciones de su sección de la fila
		mi_resultado = 0;
		for (k = 0; k < counts[mi_rango]; k++)
			mi_resultado += (seccion_fila[k] * seccion_incognitas[k]);
		
#ifdef DEBUG
		printf("[%d] Mi resultado = %.3f\n", mi_rango, mi_resultado); // TODO (quitar)
#endif

		// Envío y recepción en el root de los resultados individuales
		MPI_Gather(&mi_resultado, 1, MPI_FLOAT, resultados, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

#ifdef DEBUG
		// if (mi_rango == 0)
		// {
		// 	printf("[*] Resultados ->");
		// 	for (k = 0; k < num_procs; k++)
		// 		printf(" [%.3f]", resultados[k]);
		// 	printf("\n");
		// }
#endif

		if (mi_rango == 0)
		{
			// Cómputo de 'b'
			for (k = 0; k < num_procs; k++)
				temp -= resultados[k];

			// a * x = b -> x = b (temp) / a (pivote)
			// x[dim - ((i + 1 + dim) % dim)] = temp / A[i * (dim + 1) + i];
			x[i] = temp / A[i * (dim + 1) + i];
		}

		// Prepara los arrays de no. de eltos. y desplazamientos para la siguiente fila
		inicializa_counts(counts, num_procs, dim - i, 1, true);
		inicializa_displs(displs, num_procs, counts);
	}

#ifdef RESULTS
	if (mi_rango == 0)
	{
		printf("[0] Resultados:");
		for (k = 0; k < dim; k++)
			printf("   x[%d]: %f", k, x[k]);
		printf("\n");
	}
#endif

	MPI_Barrier(MPI_COMM_WORLD);
	if (mi_rango == 0)
	{
		t_fin = MPI_Wtime();
		printf("[0] Tiempo: %f\n", t_fin - t_inicio);
	}

	// Escritura de resultados en el fichero de salida
	if (mi_rango == 0) escribe_resultados(x, dim, filename);

	free(A);
	free(x);
	free(counts);
	free(displs);
	free(fila_referencia);
	free(bloque_resultados);
	free(seccion_fila);
	free(seccion_incognitas);
	free(resultados);

	MPI_Finalize();

	exit(EXIT_SUCCESS);
}
