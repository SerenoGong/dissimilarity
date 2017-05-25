// implementation of Manifold Regularization
// min (1/l) sum_i (f(x_i) - y_i) + (lamda_S/(l+u)^2) fLf + lamda_K norm_K(f)
// where l: num of labeled points, u: num of unlabeled points


import java.util.Arrays;
import Jama.Matrix;

public class ManifoldRegularization {
	// input
	int n; // num of points
	double[][] x; // feature vector of each point  
	boolean[] label; // whether a point is labeled (y[i] is known)
	double[] y; // true value of each point	
	
	double[] y_est; // predicted value
	double[] alpha;
	
	// coefficient parameters for Manifold Regularization formulation
	private double lambda_S;
	private double lambda_K;
	private double gaussian_K; // parameter for kernel K
	
	// estimation error
	double error;
	
	ManifoldRegularization(double[][] x1, double[] y1, boolean[] label1) {
		x = MyMatrix.copy(MyMatrix.normalize_01(x1));
		label = Arrays.copyOf(label1, label1.length);
		y = Arrays.copyOf(y1, y1.length);
		
		n = x.length;
	}
	
	void setCoefficients(double lambdaS1, double lambdaK1,double gaussianK1) {
		lambda_S = lambdaS1;
		lambda_K = lambdaK1;
		gaussian_K = gaussianK1;
	}
	
	
	private double[][] computeMatrixH() {
		double[][] matrix_H = new double[n][n];
		for (int i = 0; i < n; i++) 
			if (label[i]) matrix_H[i][i] = 1;
		return matrix_H;
	}
	
	private double kernel(double[] x1, double[] x2, double gaussianCoeff) {
		double dist = Misc.euclideanDist(x1, x2);
		return Math.exp(dist*dist*(-1)/2/gaussianCoeff/gaussianCoeff);
	}
	
	private double[][] computeMatrixK(double gaussianCoeff) {		
		double[][] matrix_K = new double[n][n];
		for (int i = 0; i < n; i++) {
			matrix_K[i][i] = 1;
			for (int j = 0; j < i; j++) {
				// compute K
				matrix_K[i][j] = kernel(x[i], x[j], gaussianCoeff);
				matrix_K[j][i] = matrix_K[i][j];
			}
		}
		return matrix_K;
	}
	
	private double[][] computeMatrixL(double[][] W) {
		// compute L
		double[][] matrix_L = new double[n][n];	
		for (int i = 0; i < n; i++) {
			for (int j = 0; j <= i; j++) {
				if (i!=j) matrix_L[i][j] = (-1) * W[i][j];
				else {
					double sum=0;
					for (int l = 0; l < n; l++) sum = sum + W[i][l];
					matrix_L[i][j] = sum;
				}
				matrix_L[j][i] = matrix_L[i][j];
			}	
		}
		return matrix_L;
	}	
	protected void run(double[][] W) {
		// W is weight matrix for the Laplacian regularization term
		
		// start predicting...
	
		// method 1:
		// Alpha = (I+(H+L)K)^{-1} HY
		// Y_est = K Alpha = K (I+(H+L)K)^{-1} HY
	
		// method 2: also resulting in the same as method 1
		// Y_est = K Alpha = (I+K(H+L))^{-1} KHY

		int num_labels = 0;
		for (boolean b : label)	if (b) num_labels++;
		
		double lambda_S1 = lambda_S * (double) num_labels / (double) (n*n);
		double lambda_K1 = lambda_K * (double) num_labels;
		
		Matrix Y = new Matrix(y, n);
		Matrix H = new Matrix(computeMatrixH());
		Matrix K = new Matrix(computeMatrixK(gaussian_K));
		Matrix L = new Matrix(computeMatrixL(W));
		
		Matrix I = Matrix.identity(n, n); // identity matrix 

//		K.print(10, 10);
		
		I = I.timesEquals(lambda_K1);
		L = L.timesEquals(lambda_S1);

		Matrix Alpha = L.plus(H).times(K).plusEquals(I).inverse().times(H).times(Y);
		Matrix Y_est = K.times(Alpha);
		

		alpha = Alpha.getColumnPackedCopy();
		y_est = Y_est.getColumnPackedCopy();
		error = Misc.euclideanDist(y_est,  y);
		
//		Y.print(10, 2);		
		
	}

	double predict(double[] x_new) {
		double y_new = 0;
		for (int i=0; i<n; i++)
			y_new += alpha[i]* kernel(x_new, x[i], gaussian_K);
		return y_new;
	}
	
	public static void main(String[] args) {
		// this is to test and show how to use this class
		
		double[][] x1 = {{1.1}, {2.2}, {1.1}, {3.3}, {4.4}};
		double[] y1 = {10, 20, 10, 30, 40};
		boolean[] label1 = {true, false, false, true, true};
		
		ManifoldRegularization mr = new ManifoldRegularization(x1, y1, label1);
		mr.setCoefficients(1, .01, 1);
		mr.run(null);
		
		System.out.println("Values (predicted ---> truth):");
		for (int i=0; i< mr.n; i++)
			System.out.println(mr.y_est[i] + "--->" + mr.y[i]);
		System.out.println("Prediction error: " + mr.error);
	}
}
