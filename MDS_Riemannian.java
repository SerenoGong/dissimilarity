import Jama.*;

abstract class MDS_Riemannian {
	// surface parameters
	double curvature; // curvature (0/+/- ==> euclidean/spherical/hyperbolic
	int m; // m-dimensional Riemannian manifold (subset of R^(m+1)-dimensional space)
	double radius; // radius of sphere/hyperbolic (apply only when k!=0)
	
	//INPUT
	double[][] F; // input dissimilarity matrix
	int n; // number of points
	
	//OUTPUT
	double[][] X; // image points in the embedding
	double embedding_distortion;
	double[][] geodesic_dist = null; // geodesic distance between X[i] and X[j]
	
	// variables for convenience
	protected  Matrix eigenV; // eigenvector matrix
	protected Matrix eigenD; // eigenvalue matrix
	protected int num_positive_eigen;
	protected int num_negative_eigen;
	protected boolean eigen_ready = false;
	
	abstract protected double angle(double[] x, double[] y);
	abstract protected double[] normalizeFitManifold(double[] x);
	abstract double[][] randomPoint(int n);
	abstract boolean embed();
	abstract protected double[] MapLOG(double[] x, double[] y);
	abstract protected double[] MapEXP(double[] x, double[] z);	
	

	MDS_Riemannian(int dim, double k) {
		m = dim-1; // m-dimensional Riemannian manifold (subset of R^(m+1)-dimensional space)
		curvature = k;
		radius = 1/Math.sqrt(Math.abs(curvature));
	}
	
	void loadDissimMatrix(double[][] F1) {
		// note: entries of F1 must have been normalized to [0, 1].
		// hence, max(F1 values) = 1.
		if (MyMatrix.max(F1) > 1 || MyMatrix.max(F1) <0) {
			System.err.println("loadDissimMatrix(): matrix F1 must have been normalized to [0, 1]");
			System.exit(-1);
		}
		F = MyMatrix.copy(F1);
		n = F.length;
	}
	

	double geodesic(double[] x, double[] y) {
		// geodesic distance on sphere	
		return angle(x, y)/Math.sqrt(Math.abs(curvature));
	}
	
	double[][] geodesic(double[][] X) {
		// geodesic distance matrix 
		int num=X.length;
		double[][] dist = new double[num][num];
		for (int i=0; i<num;i++)
			for (int j=0; j<i; j++) {
				dist[i][j] = geodesic(X[i], X[j]);
				dist[j][i] = dist[i][j];
			}
		return dist;
	}
	
	
	
	double[][] normalizeFitManifold(double[][] X) {
		double[][] newX = new double[X.length][m+1];
		for (int i=0; i < X.length; i++) newX[i] = normalizeFitManifold(X[i]);
		return newX;
	}	

	double[] randomPoint() {
		// generate a random position on the Riemannian sphere of radius r in R^d dimension 
		return randomPoint(1)[0];
	}

	void eigenProcessing() {
		// algorithm is based on EigenDecomposition
		// compute A = cos(F*sqrt(k))/k
				
		Matrix A = new Matrix(n, n);
		for (int i=0; i<n; i++) {
			for (int j=0; j<i; j++) {
				if (curvature >0) A.set(i, j, Math.cos(F[i][j]*Math.sqrt(curvature))/curvature);
				else A.set(i, j, Math.cosh(F[i][j]*Math.sqrt(-curvature))/curvature);
				A.set(j, i, A.get(i, j));
			}
		}
				
		// apply EigenDecomposition: A = V D V'
		EigenvalueDecomposition e = A.eig();
		Matrix V1 = e.getV();
		Matrix D1 = e.getD(); 
		
		// rearrange eigen matrices in descending order of eigenvalues
		int[] index_list = EigenValueIndex.sortEigen(D1);
		eigenD = new Matrix(n, n);
	    for (int i=0; i<n; i++) 
	    	 eigenD.set(i, i, D1.get(index_list[i], index_list[i]));
		
	    eigenV = new Matrix(n, n);
	    for (int row=0; row<n; row++) 
	    	 for (int column=0; column<n; column++) 
	    		 eigenV.set(row, column, V1.get(row, index_list[column]));
		
		num_positive_eigen = 0;
		num_negative_eigen = 0;
		
		for(int i=0; i<n; i++) {
			if (eigenD.get(index_list[i], index_list[i]) > 0) 
				num_positive_eigen++;
			if (eigenD.get(index_list[i], index_list[i]) < 0) 
				num_negative_eigen++;
		}
		System.out.println("signature of A = (" + 
				num_positive_eigen + "," + num_negative_eigen + "," + (n-num_positive_eigen-num_negative_eigen));
		
		eigen_ready = true;
	}
	
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
			// project each y onto tangent space
			int good_projection_count=0;
			for (int j=0; j<y.length; j++) {
				double[] ylog = MapLOG(x, y[j]);
				
				//System.err.println("x=" + MyMatrix.toStr(x) + ", y["+j+"]=" + MyMatrix.toStr(y[j]));
				//System.err.println("ylog=" + MyMatrix.toStr(ylog) + "yj' = " + MyMatrix.toStr(MapEXP(x, ylog)));
				
				good_projection_count++;
				
				// translate ylog by distances
				double g = MyMatrix.norm(ylog); // which is the same as geodesic(x, y[j], k), but faster
				
				if(Double.isNaN(g)) {
					System.err.println("multilateration(): g=NaN; report!");
					System.err.println("x=" + MyMatrix.toStr(x) + ", y["+j+"]=" + MyMatrix.toStr(y[j]));
					System.exit(-1);
				}
				
				double scale = 1.0 - distances[j]/g;

				if(Double.isNaN(scale)) {
					System.err.println("multilateration(): scale=NaN; report!");
					System.err.println("g = " + g);
					System.exit(-1);
				}
				
				if(Double.isInfinite(scale)) {
					// when g is TOO SMALL => 1/SMALL = BIG
					// JAVA cannot handle TOO BIG double numbers, hence setting scale= -INFINITE
					// when g is too small, meaning y[j] is too close to origin x
					// push farther away
					System.out.println("WARNING!!! multilateration(): scale=-INFINITE; g=" + g);
					System.err.println("...ylog=" + MyMatrix.toStr(ylog));
					System.err.println("...x=" + MyMatrix.toStr(x));
					System.err.println("...y["+j+"]=" + MyMatrix.toStr(y[j]));
					scale = -1.0;
				}
				for (int i=0; i<m1; i++) {
					ylog[i] *= scale;
					ylog_centroid[i] += ylog[i];
				}
			}
			for (int i=0; i<m1; i++) ylog_centroid[i] /= (double) good_projection_count;
			
			// project ylog_centroid back to manifold 
			// length of (ylog_centroid) should be at most MAX(geo_dist)
			// for sphere: max = PI*r
			// for hyperbolic: no limit
			
			double MAX_LEN = (curvature > 0)? Math.PI * radius-0.01 : Double.MAX_VALUE;
			
			double length = MyMatrix.norm(ylog_centroid); 
			if(Double.isNaN(length) || Double.isNaN(ylog_centroid[0])) {
				System.err.println("multilateration(): ylog_centroid=NaN error, report!");				
				System.exit(-1);
			}
			
			if (length > MAX_LEN) {
				// clip to fit
				for (int i=0; i<m1; i++) ylog_centroid[i] /= (length/MAX_LEN); 
			}
			
			double[] new_x = MapEXP(x, ylog_centroid);
			new_x = normalizeFitManifold(new_x);
			
			// quit loop early if no significant change in x
			if (Misc.euclideanDist(new_x, x) < 0.0000001) break;
			else x = new_x;			
			
			// check for possible error due to floating numbers computation
			if(Double.isNaN(x[0])) {
				System.err.println("ERROR!!! multilateration(): x[0]=NaN; report!");
				System.err.println("ylog_centroid[0]=" + ylog_centroid[0]);
				System.exit(-1);
			}
		}		
		return x;
	}
}
