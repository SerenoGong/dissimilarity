import java.util.ArrayList;
import java.util.Arrays;

public class SphereMDSPartial extends SphereMDS {
	boolean[][] W; // W[i][j]=true iff the entry dissimilarity F[i][j] is missing (not usable)
	ArrayList<ArrayList<Integer>> listW; // listW[i] is the set of nodes j such that W[i][j]=true;
	

	double train_error; // average distortion error for known dissimilarity
	double test_error; // average distortion error for unknown dissimilarity based on ground-truth info
	
	SphereMDSPartial(double[][] F1, boolean[][] W1, int m1, double k1) {
		super(m1, k1);
		loadDissimMatrix(F1);
		
		W = MyMatrix.copy(W1);
		listW = new ArrayList<ArrayList<Integer>>();
		for (int i=0; i<n; i++) {
			// list of points whose dissimilarity with i is known
			ArrayList<Integer> list_i = new ArrayList<Integer>();
			for (int j=0; j<n; j++) 
				if (W[i][j] && i!=j) list_i.add(j);
			listW.add(list_i);
			if (list_i.size() == 0) {
				System.err.println("bad matrix W: point " + i + " has no dissimilarity info");
				System.exit(-1);
			}
			if (list_i.size() == 1) {
				System.err.println("bad matrix W: point " + i + " has only 1 dissimilarity info");
				System.exit(-1);
			}
		}
	}
	
	void computeErrors() {
		geodesic_dist = geodesic(X);
		train_error = 0;
		test_error = 0;
		int count_train=0;
		int count_test=0;
		for (int i=0; i<n; i++) {
			for (int j=0; j<i; j++) {
				double diff = geodesic_dist[i][j]- F[i][j];
				if (W[i][j]) {
					train_error += diff*diff;
					count_train++;
				}
				else {
					test_error += diff*diff;
					count_test++;
				}
			}
		}
		train_error = Math.sqrt(train_error) / (double) count_train;
		test_error = Math.sqrt(test_error) / (double) count_test;
	}
	
	void embedSphere_IncompleteDissim(int num_iterations) {
		// W[i][j]=true iff F[i][j] is known
		// this algorithm is an iterative algorithm
		
		// initialization
		X = randomPoint(n);
		
		// iterative algorithm to revise X
		for (int count=0; count < num_iterations; count++) {
			double[][] X_new = new double[n][m+1];
			for (int i=0; i<n; i++) {
				// list of points whose dissimilarity with i is known
				ArrayList<Integer> list = listW.get(i);
				double[][] y = new double[list.size()][m+1];
				double[] distances = new double[list.size()];
				for (int j=0; j<list.size(); j++) {
					int j1 = list.get(j);
					y[j] = X[j1];
					distances[j] = F[i][j1];
				}
				if (list.size() ==0) X_new[i] = Arrays.copyOf(X[i], m+1);
				else {
					X_new[i] = multilateration(X[i], y, distances, 100); 
					//X_new[i] = multilateration(randomPoint(), y, distances, 100); 
				}
			}
			X = X_new;
		}
	}
	
	static SphereMDSPartial bestMDS_IncompleteDissim(double[][] F, boolean[][] W, int min_m, int max_m, int num_iterations) {
		// find the best dimension, best curvature
		SphereMDSPartial best_mds = null;
		double min = Double.MAX_VALUE;
		for (int m1=min_m; m1 <= max_m; m1++) {
			double k1 = 0.1;
			while(k1 < Math.PI*Math.PI) {
				SphereMDSPartial mds = new SphereMDSPartial(F, W, m1, k1);
				mds.embedSphere_IncompleteDissim(num_iterations);
				mds.computeErrors();
				if (mds.train_error < min) {
					min = mds.train_error;
					best_mds = mds;
				}
//				System.out.println("k1=" + k1);
				k1 += 0.1;
			}	
		}
//		System.out.println("bestMDS_IncompleteDissim(): omega=" + min + " when best m=" + (best_mds.m+1) + ", best curvature=" + best_mds.k);
		
		return best_mds;
	}
}
