import java.util.Arrays;
import java.util.Random;

import Jama.*;

public class MDS_Hyper extends MDS_Riemannian {
	
	MDS_Hyper(int dim, double curvature) {
		super(dim, curvature);
		if (curvature > 0) {
			System.err.println("sphere curvation must be negative");
			System.exit(-1);
		}
		
	}
	
	double[][] randomPoint(int n) {
		// generate n random positions on the sphere
		// NOTE: this is NOT uniformly at random
		
		double[][] X = new double[n][m+1];	
		Random rand = new Random();
		// chose a random x0 between [radius, radius+1]
		for (int i=0; i<n; i++) {
			double x0 = radius + rand.nextDouble() + 0.1; // to avoid being equal to radius
			X[i][0] = x0;
			// (x1, x2, ...xm) is a random point on sphere (center=x0, radius = sqrt(x0^2-r^2))
			double r = Math.sqrt(x0*x0 - radius*radius);
			MDS_Sphere sphere = new MDS_Sphere(m+1, 1/r/r);
			double[] point = sphere.randomPoint();
			for (int j=1; j<=m; j++) X[i][j] = point[j-1];
		}
		return X;
	}
	
	private double innerProduct_Hyper(double[] x, double[] y) {
		double inner_product = x[0]*y[0]*(-1);
		for (int i =1; i<x.length; i++) inner_product += x[i]*y[i];
		return inner_product;
	}
	
	protected double angle(double[] x, double[] y) {
		// x and y are 2 points on the hyperboloid centered at origin with curvature k)
		double v = curvature*innerProduct_Hyper(x, y);
		// theoretically, v must be >= 1!
		if (v<1) {
			// possible  error due to floating number computation
			System.err.println("warning: hyperbolicAngle(): " + v + "; v=k*inner_product_hyperbolic(x,y) must be >= 1");
			return 0;
		}
		return Misc.acosh(v);
	}
	
	protected double[] MapLOG(double[] x, double[] y) {
		// x and y are points on the hyperbolic manifold
		// project y onto the tangent space of point x
		double[] z = new double[y.length];
		
		double theta;
		double coshtheta;
		double sinhtheta;
		double v = curvature*innerProduct_Hyper(x, y);
		
		if (v<=1) {
			//System.out.println("v=" + v + ", dist2Manfold(x), dist2Manfold(y)"+distToManifold(x)+","+distToManifold(x));
			theta= 0; // y and x are extremely close
			return z;
		}
		theta = Misc.acosh(v);
		coshtheta=v;
		sinhtheta = Math.exp(theta) - v;
		for (int i=0; i<y.length; i++)
			z[i] = theta*(y[i]-x[i]*coshtheta)/sinhtheta;
		return z;
	}
	

	protected double[] MapEXP(double[] x, double[] z) {
		// inverse of MapLOG
		// project z on the tangent space of point x back to point y on the manifold
		
		/* the following property holds
		 * norm(z) = geodesic(x,y) = r*theta(x,y)
		 * therefore, theta = norm(z)/r
		*/
		double innerproduct = innerProduct_Hyper(z, z);
		if (innerproduct <= 0) {
			// theoretically impossible, but likely due to computation's numerical error
			return Arrays.copyOf(x, x.length);
		}
		
		double theta = Math.sqrt(innerproduct)/radius;
		double coshtheta, sinhtheta;

		if (theta<0.0001) {
			// z is VERY close to the origin of the log map
			// simply return origin 
			return Arrays.copyOf(x, x.length);
		}
		
		coshtheta = Math.cosh(theta);
		sinhtheta = Math.exp(theta) - coshtheta;
		double[] y = new double[z.length];
		for (int i=0; i<z.length; i++) 
			y[i] = x[i]*coshtheta + z[i]*sinhtheta/theta;
		
		return y;
	}
	
	protected double[] normalizeFitManifold(double[] x) {
		// normalize x such that x is on the hyperbolic manifold
		// to satisfy the constraint that inner_product_hyperbolic(x,x) = -r^2=1/k		

		// find a good point on the manifold to approximate x
		double[] y = new double[m+1];
		if (x[0] <= radius) {
			// return the point (r, 0,0,...0)
			y[0] = radius;
			return y;
		}
		
		// x[0] >= radius
		// y will be the on the sphere: 
		// centered at (x[0], 0, ..., 0)
		// radius = sqrt(x[0]^2-r^2)
		// closest to x
		
		y[0] = x[0];
		double r = Math.sqrt(x[0]*x[0]-radius*radius);
		// normalize (x[1], x[2], ..., x[m]) to be on this sphere
		double a=0;
		for (int coor=1; coor< m+1; coor++) a += x[coor]*x[coor];
		a = Math.sqrt(a);
		for (int coor=1; coor< m+1; coor++)	y[coor] = x[coor] * r/ a;
		return y; 
	}
	

	boolean embed() {
		// must run eigenProcessing() already
		if (!eigen_ready) {
			System.err.println("embed(): eigencomposition must be run first; but it did not");
			System.exit(-1);
		}
		
			//  choose (m+1) eigenvalues = 
		// m largest eigenvalues (must be positive)
		// (m+1)^th: smallest eigenvalue (must be non-positive)
			
		if (m > num_positive_eigen || num_positive_eigen == n) {
			System.err.println("warning: m=" + m +" TOO HIGH; should not exceed " + (num_positive_eigen));
			return false; // unsuccessful
	    }
	    
	    Matrix D1 = new Matrix(m+1, m+1);
	    for (int i=0; i<m; i++) D1.set(i, i, eigenD.get(i, i));
	    D1.set(m, m, -eigenD.get(n-1, n-1));
	    
	    Matrix V1 = new Matrix(n, m+1);
	    for (int row=0; row<n; row++) {
	    	 for (int column=0; column<m; column++) 
	    		 V1.set(row, column, eigenV.get(row, column));
	    	 V1.set(row, m, eigenV.get(row, n-1));
	    }
	     
	    // compute square root matrix of E1 (which is a diagonal matrix)
	    for (int i=0; i< m+1; i++) D1.set(i, i, Math.sqrt(D1.get(i, i)));
	     
	    Matrix solutionX = V1.times(D1); // each row of solutionX is position X[i]
	    // normalized to make sure X is on the manifold
		X = normalizeFitManifold(solutionX.getArray());
		geodesic_dist = geodesic(X);
		embedding_distortion = MyMatrix.normFrobenius(geodesic_dist, F)/Math.sqrt(n*(n-1));
		return true;		
	}
	
	
}
