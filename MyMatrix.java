import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Random;
import java.util.Vector;

import Jama.Matrix;

public class MyMatrix {
	public static double[][] normalize(double[][] matrix, double max_value) {
		// normalize such that value does not exceed max_value
		int row = matrix.length;
		int column = matrix[0].length;
		
		double max = Double.MIN_VALUE;
		for (int i=0; i< row; i++) {
			for (int j=0; j < column; j++) {
				if (matrix[i][j] > max) max = matrix[i][j];
			}
		}
		double[][] F1 = new double[row][column];
		for (int i=0; i< row; i++) {
			for (int j=0; j < column; j++) {
				F1[i][j] = matrix[i][j] * max_value / max;
			}
		}
		return F1;
	}
	
	public static double[][] normalize_01(double[][] matrix) {
		// normalize such that value is in [0, 1]
		int row = matrix.length;
		int column = matrix[0].length;
		double min = min(matrix);
		double max = max(matrix);
		double[][] F1 = new double[row][column];
		for (int i=0; i< row; i++) {
			for (int j=0; j < column; j++) {
				if (max != min)
					F1[i][j] = (matrix[i][j]-min)  / (max-min);
				else 
					F1[i][j] /= max;
			}
		}
		return F1;
	}
	
	public static double max(double[][] F) {
		int row = F.length;
		int column = F[0].length;
		double maxF = F[0][0];
		for (int i=0; i< row; i++) {
			for (int j=0; j < column; j++) {
				if (F[i][j] > maxF) maxF = F[i][j];
			}
		}
		return maxF;
	}
	public static double min(double[][] F) {
		int row = F.length;
		int column = F[0].length;
		double minF = F[0][0];
		for (int i=0; i< row; i++) {
			for (int j=0; j < column; j++) {
				if (F[i][j] < minF) minF = F[i][j];
			}
		}
		return minF;
	}
	public static double[][] copy(double[][] matrix) {
		if (matrix == null) return null;
		double[][] A = new double[matrix.length][matrix[0].length];
		for (int i=0; i < matrix.length; i++)
	    	for (int j=0; j<matrix[0].length; j++) 
	    		A[i][j] = matrix[i][j];
		return A;
	}
	
	
	public static boolean[][] copy(boolean[][] matrix) {
		if (matrix == null) return null;
		boolean[][] A = new boolean[matrix.length][matrix[0].length];
		for (int i=0; i < matrix.length; i++)
	    	for (int j=0; j<matrix[0].length; j++) 
	    		A[i][j] = matrix[i][j];
		return A;
	}
	
	public static double[][] identity(int len){
		double[][] identity_matrix = new double[len][len];
		for(int i = 0; i < len; i++) identity_matrix[i][i] = 1;
		return identity_matrix;
	}
	
	
	
	//product of two matrices
	public static double[][] multiply(double[][] matrix_a, double[][] matrix_b){
		int M = matrix_a.length;
		int N = matrix_b.length;
		int P = matrix_b[0].length;
		
		double[][] product_matrix = new double[M][P];
		
		for(int i = 0; i < M; i++){
			for(int j = 0; j < P; j++){
				for(int k = 0; k < N; k++)
					product_matrix[i][j] += matrix_a[i][k] * matrix_b[k][j];
			}
		}
		
		return product_matrix;
	}
	
	//get the sum of two matrices
	public static double[][] add(double[][] matrix_a, double[][] matrix_b){
			int M = matrix_a.length;
		int N = matrix_a[0].length;

		double[][] c = new double[M][N];
		
		for(int i = 0; i < M; i++){
			for(int j = 0; j < N; j++){
				c[i][j] = matrix_a[i][j] + matrix_b[i][j];
			}
		}
		return c;
	}
	public static double[][] subtract(double[][] matrix_a, double[][] matrix_b){
		int M = matrix_a.length;
		int N = matrix_a[0].length;

		double[][] c = new double[M][N];
		
		for(int i = 0; i < M; i++){
			for(int j = 0; j < N; j++){
				c[i][j] = matrix_a[i][j] - matrix_b[i][j];
			}
		}
		return c;
	}

	//get product of a number and a matrix
	public static double[][] multiplyScalar(double c, double[][] A){
		int M = A.length;
		int N = A[0].length;

		double[][] B = new double[M][N];
		
		for(int i = 0; i < M; i++){
			for(int j = 0; j < N; j++){
				B[i][j] = c * A[i][j];
			}
		}
		
		return B;
	}
	
	//get product of a matrix and a vector
	public static double[] multiplyVector(double x[], double[][] A){
		int M = A.length;
		int N = A[0].length;

		double[] product_vector = new double[M];
		
		for(int i = 0; i < M; i++){
			for(int j = 0; j < N; j++){
				product_vector[i] += A[i][j] * x[j];
			}
		}
		
		return product_vector;
	}
	
	public static String toStr(double[] vector){
		String res="";
		for(double tempi : vector)
			res = res + (tempi + " ");
		return res;
	}
	
	public static String toStr(double[][] matrix){
		String res="";
		for(double[] row : matrix){
			res = res + toStr(row);
			res = res +"\n";
		}
		return res;
	}
	
	
	public static void saveAs(double[][] matrix, String save_to_file){
		try {
			PrintStream ps = new PrintStream(save_to_file);
			for(double[] row : matrix){
				boolean first_value = true;
				for(double j : row){
					if (first_value) {
						ps.print(j);
						first_value = false;
					}
					else ps.print("," + j);
				}
				ps.println();
			}
			ps.close();
		}
		catch (IOException e) {
			System.out.println("saveAs() error: " + save_to_file);
			System.exit(-1);
		}
	}
	
	public static void saveAs(int[][] matrix, String save_to_file){
		try {
			PrintStream ps = new PrintStream(save_to_file);
			for(int[] row : matrix){
				boolean first_value = true;
				for(int j : row){
					if (first_value) {
						ps.print(j);
						first_value = false;
					}
					else ps.print("," + j);
				}
				ps.println();
			}
			ps.close();
		}
		catch (IOException e) {
			System.out.println("saveAs() error: " + save_to_file);
			System.exit(-1);
		}
	}
	
	public static double[][] loadFromFile(String filename) {
		double[][] matrix = null;
		ArrayList<ArrayList<Double>> F = new ArrayList<ArrayList<Double>>();
		try{
			DataInputStream in = new DataInputStream(new FileInputStream(filename));
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			String strLine;
			int rowcount=0;
			int columncount=0;
			while((strLine = br.readLine()) != null) {
				String[] s = strLine.split(",");
				ArrayList<Double> row = new ArrayList<Double>();
				columncount = s.length;
				for (int i=0; i< s.length; i++) {
					row.add(Double.parseDouble(s[i].trim()));
				}
				F.add(row);
				rowcount++;
			}
			// convert F to double[][]
			matrix = new double[rowcount][columncount];
			for (int i=0; i < rowcount; i++)
				for (int j=0; j<columncount; j++)
					matrix[i][j]=F.get(i).get(j);
			br.close();
		} 
		catch(IOException ioException){
			System.err.println("MyMatrix.loadFromFile() Error!" + filename);
			System.exit(-1);
		}
		return matrix;
	}
	
	static int[][] loadFromFile_Int(String filename) {
		int[][] matrix = null;
		ArrayList<ArrayList<Integer>> F = new ArrayList<ArrayList<Integer>>();
		try{
			DataInputStream in = new DataInputStream(new FileInputStream(filename));
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			String strLine;
			int rowcount=0;
			int columncount=0;
			while((strLine = br.readLine()) != null) {
				String[] s = strLine.split(",");
				ArrayList<Integer> row = new ArrayList<Integer>();
				columncount = s.length;
				for (int i=0; i< s.length; i++) {
					row.add(Integer.parseInt(s[i].trim()));
				}
				F.add(row);
				rowcount++;
			}
			// convert F to int[][]
			matrix = new int[rowcount][columncount];
			for (int i=0; i < rowcount; i++)
				for (int j=0; j<columncount; j++)
					matrix[i][j]=F.get(i).get(j);
			br.close();
		} 
		catch(IOException ioException){
			System.err.println("MyMatrix.loadFromFile() Error!" + filename);
			System.exit(-1);
		}
		return matrix;
	}
	
	//get transposed matrix
	public static double[][] transpose(double[][] init_matrix){
		int i, j;
		int	len1 = init_matrix.length;
		int len2 = init_matrix[0].length;
		
		double[][] trans_matrix = new double[len2][len1];
		
		for(i = 0; i < len1; i++){
			for(j = 0; j < len2; j++){
				trans_matrix[j][i] = init_matrix[i][j];
			}
		}
		return trans_matrix;
	}
	
	//get product of two vectors
	public static double innerProduct(double[] vectA, double[] vectB){

		int i;
		double product = 0;
		int len;
		len = vectA.length;
		
		for(i=0;i<len;i++)	product += vectA[i] * vectB[i];
	
		return product;
	}
	
	// get magnitude of a vector
	public static double norm(double[] vect){
		return Math.sqrt(innerProduct(vect, vect));
	}
	
	public static double[][] invert(double matrix[][]) {
		int n = matrix.length;
	    
		if (matrix[0].length!=n) {
	    	System.out.println("Matrix is not square");
	    	System.exit(-1);
	    }
		
		int row[] = new int[n];
	    int col[] = new int[n];
	    double temp[] = new double[n];
	    int hold , I_pivot , J_pivot;
	    double pivot, abs_pivot;
	    
	    double[][] A = copy(matrix);
	    
	    // set up row and column interchange vectors
	    for(int k=0; k<n; k++) {
	    	row[k] = k ;
	    	col[k] = k ;
	    }
	    // begin main reduction loop
	    for(int k=0; k<n; k++) {
	    	// find largest element for pivot
	    	pivot = A[row[k]][col[k]] ;
	    	I_pivot = k;
	    	J_pivot = k;
	    	for(int i=k; i<n; i++) {
	    		for(int j=k; j<n; j++) {
	    			abs_pivot = Math.abs(pivot) ;
	    			if(Math.abs(A[row[i]][col[j]]) > abs_pivot) {
	    				I_pivot = i ;
	    				J_pivot = j ;
	    				pivot = A[row[i]][col[j]] ;
	    			}
	    		}
	    	}
	    	if(Math.abs(pivot) ==  1.0E-20) {
	    		System.out.println("Matrix is singular: pivot = " + Math.abs(pivot));
	    		System.exit(-1);
	    	}
	    	hold = row[k];
	    	row[k]= row[I_pivot];
	    	row[I_pivot] = hold ;
	    	hold = col[k];
	    	col[k]= col[J_pivot];
	    	col[J_pivot] = hold ;
	       
	    	// reduce about pivot
	    	A[row[k]][col[k]] = 1.0 / pivot ;
	    	for(int j=0; j<n; j++) {
	    		if(j != k) A[row[k]][col[j]] = A[row[k]][col[j]] * A[row[k]][col[k]];
	    	}
	    	// inner reduction loop
	    	for(int i=0; i<n; i++) {
	    		if(k != i) {
	    			for(int j=0; j<n; j++) {
	    				if( k != j ) A[row[i]][col[j]] = A[row[i]][col[j]] - A[row[i]][col[k]] * A[row[k]][col[j]] ;
	    			}
	    			A[row[i]][col [k]] = - A[row[i]][col[k]] * A[row[k]][col[k]] ;
	    		}
	    	}
	    }
	    // end main reduction loop

	    // unscramble rows
	    for(int j=0; j<n; j++) {
	    	for(int i=0; i<n; i++) temp[col[i]] = A[row[i]][j];
	    	for(int i=0; i<n; i++) A[i][j] = temp[i] ;
	    }
	    // unscramble columns
	    for(int i=0; i<n; i++) {
	    	for(int j=0; j<n; j++) temp[row[j]] = A[i][col[j]] ;
	    	for(int j=0; j<n; j++) A[i][j] = temp[j] ;
	    }
	    return A;
	} // end invert
	
	public static double[][] selfSimilarityMatrix(double[] vec) {
		// compute the self-similarity matrix of a given vector vec
		// assuming Euclidean as distance if norm=2, and Manhattan if norm=1
		
		double[][] mat = new double[vec.length][vec.length];
		for (int i=0; i<vec.length; i++) {
			mat[i][i] = 0;
			for (int j=i+1; j<vec.length; j++) {
				mat[i][j] = Math.abs(vec[i]-vec[j]);
				mat[j][i] = mat[i][j];	
			}
		}
		return mat;
	}
	public static double[][] selfSimilarityMatrix(double[][] vec_seq, String dist) {
		// compute the self-similarity matrix of a given sequence of vectors vec_seq
		// assuming Euclidean as distance if norm=2, and Manhattan if norm=1
		int n = vec_seq.length;
		double[][] mat = new double[n][n];
		for (int i=0; i<n; i++) {
			mat[i][i] = 0;
			for (int j=i+1; j<n; j++) {
				if (dist.equals("euclidean"))	mat[i][j] = Misc.euclideanDist(vec_seq[i], vec_seq[j]);
				if (dist.equals("manhattan"))	mat[i][j] = Misc.manhattanDist(vec_seq[i], vec_seq[j]);
				mat[j][i] = mat[i][j];	
			}
		}
		return mat;
	}
	public static double[][] selfSimilarityMatrix(Vector<Double>[] vec_seq, String dist) {
		int n = vec_seq.length;
		double[][] mat = new double[n][n];
		for (int i=0; i<n; i++) {
			mat[i][i] = 0;
			for (int j=i+1; j<n; j++) {
				if (dist.equals("euclidean"))	mat[i][j] = Misc.euclideanDist(vec_seq[i], vec_seq[j]);
				if (dist.equals("manhattan"))	mat[i][j] = Misc.manhattanDist(vec_seq[i], vec_seq[j]);
				mat[j][i] = mat[i][j];	
			}
		}
		return mat;
	}
	
	public static double normFrobenius(double[][] A, double[][] B) {
		// A and B must have the same dimension
		double error =  Matrix.constructWithCopy(A).minus(Matrix.constructWithCopy(B)).normF();
		return error;
	}
	
	static int[][] random_01Matrix(int row, int column, double prob_of_1) {
		// generate a random 0-1 matrix
		Random rand = new Random();
		int[][] W = new int[row][column];
		for (int i=0; i<row; i++) 
			for(int j=0; j<column; j++) 
				W[i][j] = (rand.nextDouble() < prob_of_1)? 1: 0;
		return W;
	}
	static int[][] random_01Matrix(int n, double prob_of_1) {
		// generate a random symmetric 0-1 square matrix
		Random rand = new Random();
		int[][] W = new int[n][n];
		for (int i=0; i<n; i++) 
			for(int j=0; j<=i; j++) {
				W[i][j] = (rand.nextDouble() < prob_of_1)? 1: 0;
				W[j][i] = W[i][j];
			}
		return W;
	}
}
