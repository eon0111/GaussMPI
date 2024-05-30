#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>

int printResults (float *x, int n, char *filename) 
{
	int i;
	FILE * fp;

	if ((fp = fopen(filename, "w")) == NULL)
	{
		printf("Error: No puedo abrir el fichero %s\n", filename);
		exit(1);
	}

	for (i = 0; i < n; i++)
	{
		fprintf(fp, "x[%d] = %f\n", i, x[i]);
	}
	fclose(fp);
	return 0;
}

void generateLinearSystem(int n, float *A, float *b) 
{
	int i, j;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++)
			A[i * n + j] = (1.0 * n + (rand() % n)) / (i + j + 1);
		A[i * n + i] = (10.0 * n) / (i + i + 1);
	}

	for (i = 0; i < n; i++)
		b[i] = 1.0;
}

/**
 * @brief Muestra una matriz en la salida estándar.
 * @param matriz la matriz
 * @param dim_x el número de columnas de la matriz
 * @param dim_y el número de filas de la matriz
 */
void muestra_matriz(float *matriz, int dim_x, int dim_y) {
	printf("[*] MATRIZ:\n");

	int i, j;
	for (j = 0; j < dim_y; j++)
	{
		for (i = 0; i < dim_x; i++)
			printf("%6.2f", matriz[j * dim_x + i]);
		printf("\n");
	}
}

void solveLinearSystem(const float *A, const float *b, float *x, int n) 
{
	float *Acpy = (float *) malloc(n * n * sizeof(float));
	float *bcpy = (float *) malloc(n * sizeof(float));
	memcpy(Acpy, A, n * n * sizeof(float));
	memcpy(bcpy, b, n * sizeof(float));

	int i, j, count;
	float ratio;

	/* Gaussian Elimination */
	for (i = 0; i < (n - 1); i++) {
		// muestra_matriz(Acpy, n, n);
		for (j = (i + 1); j < n; j++) {
			/* ratio es el cociente del elto. de la columna que se está observando entre el pivote de esa columna */
			ratio = Acpy[j * n + i] / Acpy[i * n + i];	// i: columna --- j: fila
														// A[j * n][i] -> eltos. de cada columna
														// A[i * n][i] -> pivotes (eltos. en la diagonal)
			/* Ahora se hacen ceros en la fila */
			for (count = i; count < n; count++) {
				Acpy[j * n + count] -= (ratio * Acpy[i * n + count]);
			}
			bcpy[j] -= (ratio * bcpy[i]);
		}
	}
	// muestra_matriz(Acpy, n, n);

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

int main(int argc, char **argv) {

	int dim = 4;
	char *filename = "results.out";
	
	if (argc > 1) dim = atoi(argv[1]);

	int i, nerros = 0;

	float *A = (float *) malloc(dim * dim * sizeof(float));
	float *b = (float *) malloc(dim * sizeof(float));
	float *x = (float *) malloc(dim * sizeof(float));

	struct timeval tv_inicio, tv_fin;

	generateLinearSystem(dim, A, b);

	gettimeofday(&tv_inicio, NULL);
	solveLinearSystem(A, b, x, dim);
	gettimeofday(&tv_fin, NULL);

	printf("Tiempo: %f\n", tv_fin.tv_sec - tv_inicio.tv_sec + (float)(tv_fin.tv_usec - tv_inicio.tv_usec) / 1000000);

	// printf("[*] Resultados:");
	// for (int k = 0; k < dim; k++)
	// 	printf("   x[%d]: %f", k, x[k]);
	// printf("\n");

	printResults(x, dim, filename);

	return EXIT_SUCCESS;
}

