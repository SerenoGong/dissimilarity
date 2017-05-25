// implementation of Localization based on Manifold Regularization

import java.util.Arrays;

public class LocalizationMR {
	// input
	int n; // num of points
	int d; // location dimension (2D or 3D)

	int feature_dim;
	double[][] x; // feature vector of each point  
	
	
	double[][] location_true; // true value of each point	
	double[][] location_est; // predicted value
	boolean[] label; // whether a point is labeled (y[i] is known)
	int num_labels;
	
	// estimation error
	private double location_error;
	
	LocalizationMR(double[][] x1, double[][] y1, boolean[] label1) {
		x = MyMatrix.copy(x1);
		location_true = MyMatrix.copy(y1);
		
		label = Arrays.copyOf(label1, label1.length);
		num_labels = 0;
		for (boolean b: label) if (b) num_labels++;
		
		n = x.length;
		feature_dim = x[0].length;
		d = y1[0].length;

	}
	
	
	
	void run(double lambdaS1, double lambdaK1, double gaussianK1, double[][] W) {
		ManifoldRegularization mr;
		location_est = new double[n][d];
		for (int i=0; i<d; i++) {
			double[] y = new double[n];
			for (int point=0; point < n; point++)
				y[point] = location_true[point][i];
			mr = new ManifoldRegularization(x, y, label);
			mr.setCoefficients(lambdaS1, lambdaK1, gaussianK1);
			mr.run(W);
			for (int point=0; point < n; point++)
				location_est[point][i] = mr.y_est[point];
		}
	}
	
	double locationError() {
		location_error = 0;
		for (int i=0; i< n; i++)
			if (label[i] == false)
				location_error += Misc.euclideanDist(location_true[i], location_est[i]);
		location_error /= (n-num_labels);
		return location_error;
	}
	
	public static void main(String[] args) {
		// this is to test and show how to use this class
		
		double[][] x1 = {{1.1}, {2.2}, {1.1}, {3.3}, {4.4}};
		double[][] y1 = {{10,1}, {20,2}, {10,1}, {30,3}, {40,4}};
		boolean[] label1 = {true, false, false, true, true};
		
		LocalizationMR mr  = new LocalizationMR(x1, y1, label1);
		
		double[][] W = new double[x1.length][x1.length];
		double gaussianCoeff = 1;
		for (int i=0; i<x1.length; i++)
			for (int j=0; j<i; j++) {
				double dist = Misc.euclideanDist(x1[i], x1[j]);
				W[i][j] = Math.exp(dist*dist*(-1)/2/gaussianCoeff/gaussianCoeff);
				W[j][i] = W[i][j];
			}
		
		mr.run(1, .00001, 1, W);
		
		System.out.println(MyMatrix.toStr(mr.location_true));
		System.out.println(MyMatrix.toStr(mr.location_est));
		
		
	}
}
