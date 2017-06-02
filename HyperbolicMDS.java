import java.util.Arrays;
import java.util.Random;

import Jama.*;

public class HyperbolicMDS {
	// surface parameters
	double k; // curvature (0/+/- ==> euclidean/spherical/hyperbolic
	int m; // m-dimensional Riemannian manifold (subset of R^(m+1)-dimensional space)
	double r; // radius of sphere/hyperbolic (apply only when k!=0)
		
	//INPUT
	double[][] F; // input dissimilarity matrix
	int n; // number of points
	
	//OUTPUT
	double[][] X; // image points in the embedding
	double embedding_distortion;
	double[][] geodesic_dist; // geodesic distance between X[i] and X[j]
	
	
	HyperbolicMDS(int m1, double k1) {
		m = m1-1; // m-dimensional Riemannian manifold (subset of R^(m+1)-dimensional space)
		k = k1;
		r = 1/Math.sqrt(Math.abs(k));
	}
	
	void loadDissimMatrix(double[][] F1) {
		F = MyMatrix.copy(F1);
		n = F.length;
	}
	

	
	static double geodesic(double[] x, double[] y, double k) {
		// geodesic distance in embedding space
		// non-euclidean
		
		double inner_product = (k>=0)? x[0]*y[0] : x[0]*y[0]*(-1);
		for (int i =1; i<x.length; i++) inner_product += x[i]*y[i];
		
		double v = k*inner_product;
		
		if (k>0) {
			// v must not exceed 1 for acos() to work
			// check for possible error
			if(v>1) {
				// possible  error due to floating number computation
				System.err.println("warning: geodesic(): " + v + "; v=k*innerProduct(x,y) must not exceed 1");
				v=1;
				return 0;
			}
			return Math.acos(v)/Math.sqrt(k);
		}
		else return acosh(v)/Math.sqrt(-k);
	}
	
	
	public double[][] geodesic(double[][] X) {
		// geodesic distance matrix 
		int num=X.length;
		double[][] mat = new double[num][num];
		for (int i=0; i<num;i++)
			for (int j=0; j<i; j++) {
				mat[i][j] = geodesic(X[i], X[j], k);
				mat[j][i] = mat[i][j];
			}
		return mat;
	}

	private double[] normalizeFitManifold(double[] x) {
		// normalize x such that x is on the manifold
		// to satisfy the constraint that innerProduct(x,x)=r^2=1/k
		double[] y = new double[m+1];
		double a = innerProduct(x, x);	
		for (int coor=0; coor< m+1; coor++) 
			y[coor] = x[coor] / Math.sqrt(k*a);
		return y;
	}
	
	private double[][] normalizeFitManifold(double[][] X) {
		double[][] newX = new double[X.length][m+1];
		for (int i=0; i < X.length; i++) newX[i] = normalizeFitManifold(X[i]);
		return newX;
	}
	
	double[][] randomPoint(int n) {
		// generate n random positions X on the m-dim Riemannian surface 
		int m1 = m+1;
		double[][] X = new double[n][m1];
		if (k<0) {
			// the following will be revised to generate random positions
			// hyperbolic
			// TO BE ADDED
		}
		else {
			// spherical
			Random rand = new Random();
			double[] c = new double[m1];
			for (int i=0; i<n; i++) {
				double s=0;
				for (int j=0; j<m1; j++) {
					c[j] = rand.nextGaussian();
					s += c[j]*c[j];
				}
				for (int j=0; j<m1; j++) 
					X[i][j] = r * c[j]/ Math.sqrt(s);
			}
		}
		
		return X;
	}
	
	double[] randomPoint() {
		// generate a random position on the m-dim Riemannian surface 
		int m1 = m+1;
		double[] x = new double[m1];
		if (k<0) {
			// the following will be revised to generate random positions
			// hyperbolic
			// TO BE ADDED
		}
		else {
			// spherical
			Random rand = new Random();
			double[] c = new double[m1];
			double s=0;
			for (int j=0; j<m1; j++) {
				c[j] = rand.nextGaussian();
				s += c[j]*c[j];
			}
			for (int j=0; j<m1; j++) 
				x[j] = r * c[j]/ Math.sqrt(s);
		}
		
		return x;
	}
	
	boolean embedSphere() {
		// this algorithm works only for a complete matrix F
		// algorithm is based on EigenDecomposition
		// compute A = cos(F*sqrt(k))/k
		Matrix A = new Matrix(n, n);
		for (int i=0; i<n; i++) {
			for (int j=0; j<i; j++) {
				if (k > 0)	A.set(i, j, Math.cos(F[i][j]*Math.sqrt(k))/k);
				else A.set(i, j, Math.cosh(F[i][j]*Math.sqrt(-k))/k);
				A.set(j, i, A.get(i, j));
			}
		}
				
		// apply EigenDecomposition: A = U E U'
		EigenvalueDecomposition e = A.eig();
		
		Matrix U, E; // eigendecomposition as U E U' (U is eigenvectors, E is the diagonal matrix)
		U = e.getV();
		E = e.getD(); 
		
		// rearrange eigen matrices in descending order of eigenvalues
		int[] index_list = EigenValueIndex.sortEigen(E);
		
		int num_positive_eigen = 0;
		for(int i=0; i<n; i++) {
			if (E.get(index_list[i], index_list[i]) > 0) 
				num_positive_eigen++;
		}
		
		// only extract the part of largest eigenvalues that are positive
	    if (m+1 > num_positive_eigen) {
	    	 //System.err.println("Riemannian dimension=" + m + " too high for curvature=" + k + "; m should be at most " + (num_positive_eigen-1));
	    	 return false; // unsuccessful
	    }
	    
		// extract the part corresponding to m+1 largest (positive) eigenvalues
	    int m1=m+1;
	    Matrix E1 = new Matrix(m1, m1);
	    for (int i=0; i<m1; i++) E1.set(i, i, E.get(index_list[i], index_list[i]));
	     
	    Matrix U1 = new Matrix(n, m1);
	    for (int row=0; row<n; row++) 
	    	 for (int column=0; column<m1; column++) 
	    		 U1.set(row, column, U.get(row, index_list[column]));
	     
	    // compute square root matrix of E1 (which is a diagonal matrix)
	    for (int i=0; i< m1; i++) E1.set(i, i, Math.sqrt(E1.get(i, i)));
	     
	    Matrix solutionX = U1.times(E1); // each row of solutionX is position X[i]
	    // normalized to make sure X is on the manifold
		X = normalizeFitManifold(solutionX.getArray());
		geodesic_dist = geodesic(X);
		embedding_distortion = MyMatrix.normFrobenius(geodesic_dist, F)/Math.sqrt(n*(n-1));
		return true;
	}
	
	private double sphericalAngle(double[] x, double[] y) {
		// x and y are two vectors (2 points on the sphere centered at origin with radius r)
		double v = k*innerProduct(x, y);
		if (v>=1) return 0;
		if (v<=-1) return Math.PI;
		return Math.acos(v);
	}
	private double[] MapLOG(double[] x, double[] y) {
		// x and y are two vectors (2 points on the sphere centered at origin)
		// project y onto the tangent space of point x
		double[] z = new double[y.length];
		
		double theta;
		double costheta;
		double sintheta;
		double v = k*innerProduct(x, y);
		if (v>=1) {
			theta= 0; // y and x are extremely close
			return z;
		}
		if (v<=-1) {
			// this case should not happen; y and x are opposite on their great circle
			theta= Math.PI;
			return null;
		}

		theta = Math.acos(v);
		costheta=v;
		sintheta = Math.sqrt(1-v*v);

		for (int i=0; i<y.length; i++)
			z[i] = theta*(y[i]-x[i]*costheta)/sintheta;
		return z;
	}
	
	private double[] MapEXP(double[] x, double[] z) {
		// inverse of MapLOG
		// project z on the tangent space of point x back to point y on the sphere
		
		/* the following property holds
		 * norm(z) = geodesic(x,y) = r*theta(x,y)
		 * therefore, theta = norm(z)/r
		*/
		double theta = MyMatrix.norm(z)/r;
		double costheta, sintheta;

		if (theta<0.01) {
			// z is the origin of the log map
			return Arrays.copyOf(x, x.length);
		}
		
		if (theta > Math.PI) {
			// this should not happen
			System.err.println("MapEXP(): theta=" + theta + " > PI; warning!");
			theta = Math.PI;
			costheta=-1;
			sintheta=0;
		}
		else {
			costheta = Math.cos(theta);
			sintheta = Math.sqrt(1-costheta*costheta);
		}
		
		double[] y = new double[z.length];
		for (int i=0; i<z.length; i++) 
			y[i] = x[i]*costheta + z[i]*sintheta/theta;
		
		return y;
	}
	
	/*
	double[] multilateration(double[] x_init, double[][] y, double[] distances) {
		// given points on the sphere of radius, find the point that is a geodesic-distance away for each of those points
		// algorithm: 
		// project the points on the tangent space
		// do multilateration on the tangent space
		// last, find the point on the manifold corresponding to the multilateration point on the tangent space		
		
		// requirement: dimension of y must be (y.length) x (m+1)
		
		int m1 = m+1;
		double[] x = x_init;

		while(true) {
			double[] ylog_centroid = new double[m1];
			// project y onto tangent space
			for (int j=0; j<y.length; j++) {
				double[] ylog = MapLOG(x, y[j]);					
				// translate ylog by distances
				double g = geodesic(x, y[j], k);
				double scale = 1.0 - distances[j]/g;

				if(Double.isInfinite(scale)) {
					// when g is TOO SMALL ==> scale TOO BIG
					// JAVA cannot handle TOO BIG double numbers, hence setting scale= -INFINITE
					// when g is too small, meaning y[j] is close to origin x
					
					scale = -1.0;
				}
				for (int i=0; i<m1; i++) {
					ylog[i] *= scale;
					ylog_centroid[i] += ylog[i]/(double) y.length;
				}
			}
			
			// project ylog_centroid to sphere 
			// if ylog_centroid is too far away from origin on the tangen plane, cannot map back to sphere
			double length = MyMatrix.norm(ylog_centroid); 
			if (length > Math.PI*r) {
				// clip to fit
				for (int i=0; i<m1; i++) ylog_centroid[i] /= (length/Math.PI/r); 
				
			}
			double[] new_x = MapEXP(x, ylog_centroid);
			new_x = normalizeFitManifold(new_x);
			
			// quit loop if no significant change in x
			if (Misc.euclideanDist(new_x, x) < 0.0000001) break;
			else x = new_x;				
		}
		
		// check for possible error due to floating numbers computation
		if(Double.isNaN(x[0])) {
			System.err.println("report this error to duc.tran@umb.edu");
			System.exit(-1);
		}
		return x;
	}
	
	*/
	
	
	double[] multilateration(double[] x_init, double[][] y, double[] distances, int num_iterations) {
		// given points on the sphere of radius, find the point that is a geodesic-distance away for each of those points
		// algorithm: 
		// project the points on the tangent space
		// do multilateration on the tangent space
		// last, find the point on the manifold corresponding to the multilateration point on the tangent space		
		
		// requirement: dimension of y must be (y.length) x (m+1)
		
		int m1 = m+1;
		double[] x = x_init;

		for (int count=0; count < num_iterations; count++) {
			double[] ylog_centroid = new double[m1];
			// project y onto tangent space
			for (int j=0; j<y.length; j++) {
				double[] ylog = MapLOG(x, y[j]);					
				
				
				// translate ylog by distances
				double g = geodesic(x, y[j], k);
				double scale = 1.0 - distances[j]/g;

				if(Double.isNaN(scale)) {
					System.err.println("multilateration(): scale=NaN error, report!");
					System.err.println("g = " + g);
					System.exit(-1);
				}
				
				if(Double.isInfinite(scale)) {
					// when g is TOO SMALL ==> scale TOO BIG
					// JAVA cannot handle TOO BIG double numbers, hence setting scale= -INFINITE
					// when g is too small, meaning y[j] is close to origin x
					
					scale = -1.0;
				}
				for (int i=0; i<m1; i++) {
					ylog[i] *= scale;
					ylog_centroid[i] += ylog[i]/(double) y.length;
				}
			}
			
			// project ylog_centroid to sphere 
			// if ylog_centroid is too far away from origin on the tangen plane, cannot map back to sphere
			double length = MyMatrix.norm(ylog_centroid); 
			if(Double.isNaN(length)) {
				System.err.println("multilateration(): ylog_centroid=NaN error, report!");				
				System.exit(-1);
			}

			if (length > Math.PI*r) {
				// clip to fit
				for (int i=0; i<m1; i++) ylog_centroid[i] /= (length/Math.PI/r); 
				
			}
			double[] new_x = MapEXP(x, ylog_centroid);
			new_x = normalizeFitManifold(new_x);
			
			// quit loop early if no significant change in x
			if (Misc.euclideanDist(new_x, x) < 0.0000001) break;
			else x = new_x;			
			
			// check for possible error due to floating numbers computation
			if(Double.isNaN(x[0])) {
				System.err.println("multilateration(): NaN error, report!");
				System.exit(-1);
			}
		}
		
		
		return x;
	}
	
	static RiemannianMDS bestMDS(double[][] F1, int min_m, int max_m) {
		// find the best dimension, best curvature
		RiemannianMDS best_mds = null;
		double min = Double.MAX_VALUE;
		for (int m1=min_m; m1 <= max_m; m1++) {
			double k1 = 0.1;
			while(k1 < Math.PI*Math.PI) {
				RiemannianMDS mds = new RiemannianMDS(m1, k1);
				mds.loadDissimMatrix(F1);
				if (mds.embedSphere()) {
					if (mds.embedding_distortion < min) {
						min = mds.embedding_distortion;
						best_mds = mds;
					}
				}
				k1 += 0.1;
			}	
		}
		return best_mds;
	}
	
}
