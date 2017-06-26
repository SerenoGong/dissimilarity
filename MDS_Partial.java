import java.util.ArrayList;
import java.util.Arrays;

public class MDS_Partial {
	MDS_Riemannian mds;
	int n;
	int m; // Riemannian-dimension
	
	boolean[][] W; // W[i][j]=true iff the entry dissimilarity F[i][j] is missing (not usable)
	private ArrayList<ArrayList<Integer>> listW; // listW[i] is the set of nodes j such that W[i][j]=true;
	

	double train_error; // average distortion error for known dissimilarity
	double test_error; // average distortion error for unknown dissimilarity based on ground-truth info
	
	MDS_Partial(double[][] F1, boolean[][] W1, int m1, double k1) {
		if (k1 >0) {
			// spherical embedding
			mds = new MDS_Sphere(m1, k1);
		}
		else if (k1 <0) {
			// hyperbolic embedding
			mds = new MDS_Hyper(m1, k1);
		}
		else {
			System.out.println("MDS_Partial(): curvature k cannot be zero");
			System.exit(-1);
		}
		
		mds.loadDissimMatrix(F1);
		n = mds.n;
		m = mds.m;
		
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
		double[][] geodesic_dist = mds.geodesic_dist;
		train_error = 0;
		test_error = 0;
		int count_train=0;
		int count_test=0;
		for (int i=0; i<n; i++) {
			for (int j=0; j<i; j++) {
				double diff = geodesic_dist[i][j]- mds.F[i][j];
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
	
	void embed(int num_iterations) {
		// W[i][j]=true iff F[i][j] is known
		// this algorithm is an iterative algorithm
		
		// initialization
		double[][] X = mds.randomPoint(n);
		double[][] F = mds.F;
		
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
				else 
					X_new[i] = mds.multilateration(X[i], y, distances, 100); // should be better than random init point
			}
			X = X_new;
		}
		mds.X = X;
		mds.geodesic_dist = mds.geodesic(X);
	}
	
	static MDS_Partial bestMDS(
			double[][] F, boolean[][] W, 
			int m_min, int m_max, 
			double k_min, double k_max, double k_step,	
			int num_iterations) {
		// find the best dimension, best curvature
		MDS_Partial best_mds = null;
		double min = Double.MAX_VALUE;
		for (int m=m_min; m <= m_max; m++) {
			double k = k_min;
			while(k <= k_max) {
				if (k!=0) { 
					MDS_Partial mds = new MDS_Partial(F, W, m, k);
					mds.embed(num_iterations);
					mds.computeErrors();
					if (mds.train_error < min) {
						min = mds.train_error;
						best_mds = mds;
					}
//				System.out.println("curvature k=" + k);		
				}				
				k += k_step;
			}	
		}
//		System.out.println("bestMDS_IncompleteDissim(): omega=" + min + " when best m=" + (best_mds.m+1) + ", best curvature=" + best_mds.k);
		
		return best_mds;
	}
}
