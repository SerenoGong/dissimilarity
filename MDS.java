// IMPLEMENTATION OF CLASSICAL MULTIDIMENSIONAL SCALING

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import Jama.*;

public class MDS {
	int n; // num of points
	double[][] F; // input dissim matrix
	
	int m; // embedding dimension
	double[][] X; // image points in the embedding

	double train_err;
	double test_err;
	
	MDS(double[][] F1) {
		F = MyMatrix.copy(F1);
		n = F.length;	
	}

	void setDimension(int m1) {
		m = m1;
	}
	
	boolean embed() {
		// assuming a complete matrix F
		
		
		// compute D2 = square matrix of F
		double[][] D2 = new double[n][n];
		for (int i=0; i< n; i++) {
			for (int j=0; j < i; j++) {
				D2[i][j] = F[i][j]*F[i][j];
				D2[j][i] = D2[i][j];
			}
		}
		
		// C = centering matrix
		double[][] C = new double[n][n];
		for (int i=0; i< n; i++) {
			C[i][i] = 1.0 - 1.0 / (double) (n);
			for (int j=0; j < i; j++) {
				C[i][j] = -1.0/(double)n;
				C[j][i] = C[i][j];
			}
		}
		
		
		// B =-1/2 * C * D2 * C
		Matrix mC = new Matrix(C);
		Matrix mB = mC.times(-0.5);
		mB = mB.times(new Matrix(D2));
		mB = mB.times(mC);

		EigenvalueDecomposition e = mB.eig();
		Matrix U = e.getV();
		Matrix E = e.getD(); 
		
		//U.print(10, 2);
		//E.print(10, 2);

		// rearrange eigen matrices in descending order of eigenvalues
		int[] index_list = EigenValueIndex.sortEigen(E);
		
		int num_positive_eigen = 0;
		for(int i=0; i<n; i++) {
			if (E.get(index_list[i], index_list[i]) > 0) 
				num_positive_eigen++;
		}
		
		// only extract the part of largest eigenvalues that are positive
	    if (m > num_positive_eigen) {
	    	 System.out.println(" m= " + m + " too high; m should be at most  " + num_positive_eigen);
	    	 return false; // unsuccessful
	    }
	     
	    // extract the part corresponding to m largest (positive) eigenvalues
	    Matrix E1 = new Matrix(m, m);
	    for (int i=0; i<m; i++) 
	    	 E1.set(i, i, E.get(index_list[i], index_list[i]));
	     
	    Matrix U1 = new Matrix(n, m);
	     for (int row=0; row<n; row++) 
	    	 for (int column=0; column<m; column++) 
	    		 U1.set(row, column, U.get(row, index_list[column]));
	     
	     
	    //U1.print(10, 2);
		//E1.print(10, 2);
			
	     // compute square root matrix of E1
	     for (int i=0; i<m; i++)
	    	 E1.set(i, i, Math.sqrt(E1.get(i, i)));
	     
	    Matrix solutionX = U1.times(E1); // each row of solutionX is position X[i]
	    X = solutionX.getArray();
		train_err = MyMatrix.normFrobenius(distance(X), F)/Math.sqrt(n*(n-1));
		return true;
	}
	
	double[][] distance(double[][] X) {
		//  distance matrix 
		int n=X.length;
		double[][] mat = new double[n][n];
		for (int i=0; i<n;i++)
			for (int j=0; j<i; j++) {
				mat[i][j] = Misc.euclideanDist(X[i], X[j]);
				mat[j][i] = mat[i][j];
			}
		return mat;
	}

	double[] multilateration(double[] x_init, double[][] y, double[] distances) {
		// given points on the sphere of radius, find the point that is a geodesic-distance away for each of those points
		// algorithm: 
		// project the points on the tangent space
		// do multilateration on the tangent space
		// last, find the point on the manifold corresponding to the multilateration point on the tangent space		
		
		// requirement: dimension of y must be (y.length) x (m+1)
		
		double[] x = x_init;
		
		
		while(true) {
			double[] new_x = new double[m];
			for (int j=0; j<y.length; j++) {
				double dist = Misc.euclideanDist(x, y[j]);
				double scale = (1.0 - distances[j]/dist);
				
				if(Double.isInfinite(scale)) {
					// this case is possible when dist is TOO SMALL ==> scale TOO BIG
					// JAVA cannot handle TOO BIG double numbers, hence setting scale= -INFINITE
					// to resolve this, we simply set scale to a smaller number					
					scale = -1.0;
				}
				
				for (int i=0; i<m; i++) 
					new_x[i] += (x[i] + (y[j][i]-x[i])*scale)/(double) y.length;
			}
			
			
			// quit loop if no significant change in x
			if (Misc.euclideanDist(new_x, x) < 0.00000001) break;
			else x = new_x;
		}
		return x;
	}
	static MDS bestMDS(double[][] F1, int min_m, int max_m) {
		// find the best dimension, best curvature
		MDS best_mds = null;
		double min = Double.MAX_VALUE;
		
		for (int m1=min_m; m1 <= max_m; m1++) {
			MDS mds = new MDS(F1);
			mds.setDimension(m1);
			if (mds.embed()) {
				if (mds.train_err < min) {
					min = mds.train_err;
					best_mds = mds;
				}
			}
		}
		return best_mds;
	}
	void embed_IncompleteDissim(boolean[][] W, int num_iterations) {
		// W[i][j]=true iff F[i][j] is known
		// this algorithm is an iterative algorithm
		
		ArrayList<ArrayList<Integer>> listW = new ArrayList<ArrayList<Integer>>();
		for (int i=0; i<n; i++) {
			ArrayList<Integer> list = new ArrayList<Integer>();
			for (int j=0; j<n; j++) 
				if (W[i][j] && i!=j) list.add(j);
			listW.add(list);
		}
		
		// initialization; make sure X cannot be the origin point, or else problem: may LOOP forever
		X = new double[n][m]; 
		for (int i=0; i<n; i++)
			for (int j=0; j<m; j++)
				X[i][j] = Math.random();
		
		
		// iterative algorithm to revise X
		for (int count=0; count < num_iterations; count++) {
			double[][] X_new = new double[n][m];
			for (int i=0; i<n; i++) {
				// list of points whose dissimilarity with i is known
				ArrayList<Integer> list = listW.get(i);
				
				double[][] y = new double[list.size()][m];
				double[] distances = new double[list.size()];
				for (int j=0; j<list.size(); j++) {
					y[j] = X[list.get(j)];
					distances[j] = F[i][list.get(j)];
				}
				if (list.size()==0) X_new[i] = Arrays.copyOf(X[i], m);
				else X_new[i] = multilateration(X[i], y, distances);
			}
			X = X_new;
		}
		train_err = 0;
		test_err = 0;
		int train_count=0;
		int test_count=0;
		for (int i=0; i<n; i++) {
			for(int j=0; j<i; j++) {
				double diff = Misc.euclideanDist(X[i], X[j])-F[i][j];
				if (W[i][j]) {
					train_count++;
					train_err += diff*diff;
				}
				else {
					test_count++;
					test_err += diff*diff;
				}
			}
		}
		train_err = Math.sqrt(train_err)/(double) train_count;
		test_err = Math.sqrt(test_err)/(double) test_count;
		System.out.println("train err=" + train_err + ", test_err=" + test_err);	
	}
}

class EigenValueIndex implements Comparable<EigenValueIndex> {
	int index;
	Double value;

	public EigenValueIndex(int index, double value) {
		this.index = index; this.value = value;
	}

	@Override
	public int compareTo(EigenValueIndex o) { 
		return o.value.compareTo(value); 
	}
	
	public static int[] sortEigen(Matrix E) {
		// E is a diagonal matrix
		// sort E in descending order of eigenvalues (diagonal values of E)
		int n = E.getRowDimension();
		// output the indices in increasing order of eigen values
		int[] index_list = new int[n];
		
		ArrayList<EigenValueIndex> eigenValueIndices = new ArrayList<EigenValueIndex>();
		for(int i=0; i<n; i++) eigenValueIndices.add(new EigenValueIndex(i, E.get(i, i)));
		Collections.sort(eigenValueIndices); // increasing value order
		
		int row = 0;
	    for (EigenValueIndex eigen: eigenValueIndices) {
	    	index_list[row] = eigen.index;
	    	row++;
	    }
	    return index_list;
	}
};