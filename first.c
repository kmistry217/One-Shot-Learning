#include<stdio.h>
#include<stdlib.h>

int main (int argc, char** argv){

	char* trainingFile = argv[1];
	FILE* fp = NULL;
	fp = fopen(trainingFile, "r");
	
	int K, N; //number of attributes and training examples
	double** X,**XT,**XTX,**inverse,**IXT; 
	double* Y, *W, *P;  
	int r,c,k,a,b,d;
	double sum = 0, temp, temp2; 
	
	fscanf(fp, "%d\n", &K);
	fscanf(fp, "%d\n", &N);
	
	
	X = (double**) malloc (N * sizeof(double*));  
	for (r=0; r<N; r++){
		X[r] = (double*) malloc ((K+1) * sizeof(double));
	}
	XT = (double**) malloc ((K+1) * sizeof(double*));  
	IXT = (double**) malloc ((K+1) * sizeof(double*));  
	for (r=0; r<(K+1); r++){
		XT[r] = (double*) malloc ((N) * sizeof(double));
		IXT[r] = (double*) malloc ((N) * sizeof(double));
	}
	XTX = (double**) malloc ((K+1) * sizeof(double*)); 
	inverse = (double**) malloc ((K+1) * sizeof(double*));
	for (r=0; r<(K+1); r++){
		XTX[r] = (double*) malloc ((K+1) * sizeof(double));
		inverse[r] = (double*) malloc ((K+1) * sizeof(double));
	}
	Y = (double*) malloc (N * sizeof(double));  
	W = (double*) malloc ((K+1) * sizeof(double));  
	
		
	for (r=0; r<N; r++){   //scans matrix in 
		X[r][0] = 1;
		for (c=1; c<K+1; c++){
			fscanf(fp, "%lf,", &X[r][c]);
		}
		fscanf(fp, "%lf\n", &Y[r]);
	}
	
	fclose(fp);
	
	
	char* testFile = argv[2];
	FILE* fp2 = NULL;
	fp2 = fopen(testFile, "r");
	
	int M; 
	double **test;
	
	fscanf(fp2, "%d\n", &M);
	
	test = (double**) malloc (M * sizeof(double*));  
	for (r=0; r<M; r++){
		test[r] = (double*) malloc ((K+1) * sizeof(double));
	}
	P = (double*) malloc (M * sizeof(double));
	
	for (r=0; r<M; r++){   //scans test matrix 
		test[r][0] = 1;
		for (c=1; c<K+1; c++){
			fscanf(fp2, "%lf,", &test[r][c]);
		}
	}

	fclose(fp2);
	
	
	for (r=0; r<(K+1); r++){ // transposes matrix
		for (c=0; c<(N); c++){
			XT[r][c] = X[c][r];
		}
	}
	
	for (r=0; r<(K+1); r++){ //multiplies tranposed by original
		for (c=0; c<(K+1); c++){
			for (k=0; k<N; k++){
				sum = sum + XT[r][k] * X[k][c];
			}
			XTX[r][c] = sum;
			sum = 0; 
		}
	}
	
	for (r=0; r<(K+1); r++){ //sets up identity matrix
		for (c=0; c<(K+1); c++){
			if (r == c){
				inverse[r][c] = 1;
			}
			else{
				inverse[r][c] = 0;
			}
		}
	}
	
	for (a=0; a<(K+1); a++){ //gaussian elimination forward
		temp = XTX[a][a];
		for (b=0; b<(K+1); b++){
			inverse[a][b] = inverse[a][b] / temp;
			XTX[a][b] = XTX[a][b] / temp;
		}
		for (d=a+1; d<(K+1); d++){
			temp2 = XTX[d][a];
			for (c=0; c<(K+1); c++){
				inverse[d][c] = inverse[d][c] - (inverse[a][c] * temp2);
				XTX[d][c] = XTX[d][c] - (XTX[a][c] * temp2);
			}
		}
	}
	for (a=K; a>0; a--){  //gaussian elimination backward
		for (b=a-1; b>=0; b--){
			temp = XTX[b][a];
			for (c=K; c>=0; c--){
				inverse[b][c] = inverse[b][c] - (inverse[a][c] * temp);
				XTX[b][c] = XTX[b][c] - (XTX[a][c] * temp);
			}
		}
	}
	
	for (r=0; r<(K+1); r++){ //multiplies inverse by transposed
		for (c=0; c<N; c++){
			for (k=0; k<(K+1); k++){
				sum = sum + inverse[r][k] * XT[k][c];
			}
			IXT[r][c] = sum;
			sum = 0; 
		}
	}
	
	for (r=0; r<(K+1); r++){ //finds W by multiplying IXT x Y
		for (k=0; k<N; k++){
			sum = sum + IXT[r][k] * Y[k];
		}
		W[r] = sum;
		sum = 0; 
	}
	
	for (r=0; r<M; r++){ //finds prediction prices
		for (k=0; k<(K+1); k++){
			sum = sum + test[r][k] * W[k];
		}
		P[r] = sum;
		sum = 0; 
	}
	
	
	for (r=0; r<M; r++){   //prints prediction prices
		printf ("%0.0lf\n", P[r]);
	}

	for(r=0; r<N; r++){ //free allocated memory for the matrixes
		free(X[r]);
	}
	free(X);
	for(r=0; r<(K+1); r++){
		free(XT[r]);
		free(IXT[r]);
		free(XTX[r]);
		free(inverse[r]);
	}
	free(XT);
	free(IXT);
	free(XTX);
	free(inverse);
	for(r=0; r<M; r++){
		free(test[r]);
	}
	free(test);
	free(Y);
	free(W);
	free(P);
		
	return 0; 
}
