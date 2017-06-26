import java.util.Arrays;
import java.util.Random;

import Jama.*;

public class MDS_Sphere extends MDS_Riemannian {
	
	MDS_Sphere(int dim, double curvature) {
		super(dim, curvature);
		if (curvature <= 0) {
			System.err.println("sphere curvation must be positive");
			System.exit(-1);
		}
		
	}
	
	
	protected double angle(double[] x, double[] y) {
		// x and y are two vectors (2 points on the sphere centered at origin with radius r)
		double v = curvature*MyMatrix.innerProduct(x, y);
		if (v>1) {
			// possible  error due to floating number computation
			System.err.println("warning: sphericalAngle(): " + v + "; v=k*innerProduct(x,y) must be <= 1");
			return 0;
		}
		if (v<-1) {
			// possible  error due to floating number computation
			System.err.println("warning: sphericalAngle(): " + v + "; v=k*innerProduct(x,y) must be >= -1");
			return Math.PI;
		}
		return Math.acos(v);
	}
	
	protected double[] normalizeFitManifold(double[] x) {
		// normalize x such that x is on the manifold
		// to satisfy the constraint that innerProduct(x,x)=r^2=1/k
		double[] y = new double[m+1];
		double a = MyMatrix.norm(x)/radius;	
		for (int coor=0; coor< m+1; coor++) y[coor] = x[coor] / a;
		return y;
	}
	
	double[][] randomPoint(int n) {
		// generate n random positions on the sphere
		// uniformly at random
		int dim = m+1;
		double[][] X = new double[n][dim];
		Random rand = new Random();
		double[] c = new double[dim];
		for (int i=0; i<n; i++) {
			double s=0;
			for (int j=0; j<dim; j++) {
				c[j] = rand.nextGaussian();
				s += c[j]*c[j];
			}
			for (int j=0; j<dim; j++) 
				X[i][j] = radius * c[j]/ Math.sqrt(s);
		}
		return X;
	}
	
	public boolean embed() {
		// must run eigenProcessing() already
		if (!eigen_ready) {
			System.err.println("embed(): eigencomposition must be run first; but it did not");
			System.exit(-1);
		}
		
		//  choose (m+1) eigenvalues = (m+1) largest eigenvalues (ideally, should be positive)			
		if (m+1 > num_positive_eigen) {
	    	 System.err.println("warning: m=" + m +" TOO HIGH; should not exceed " + (num_positive_eigen-1));
	    	 return false; // unsuccessful
	    }
	    
		// extract the part corresponding to m+1 largest (positive) eigenvalues
	    Matrix D1 = eigenD.getMatrix(0, m, 0, m);
	    Matrix V1 = eigenV.getMatrix(0, n-1, 0, m);
	     
	    // compute square root matrix of D1 (which is a diagonal matrix)
	    for (int i=0; i< m+1; i++) D1.set(i, i, Math.sqrt(D1.get(i, i)));
	     
	    Matrix solutionX = V1.times(D1); // each row of solutionX is position X[i]
	    // normalized to make sure X is on the manifold
		X = normalizeFitManifold(solutionX.getArray());
		geodesic_dist = geodesic(X);
		embedding_distortion = MyMatrix.normFrobenius(geodesic_dist, F)/Math.sqrt(n*(n-1));
		return true;
	}
	
	
	protected double[] MapLOG(double[] x, double[] y) {
		// x and y are two points on the spherical manifold
		// project y onto the tangent space of point x
		double[] z = new double[y.length];
		
		double theta;
		double costheta;
		double sintheta;
		
		double v = curvature*MyMatrix.innerProduct(x, y);
		
		if (v>=1) {
			theta= 0; // y and x are extremely close
			return z;
		}
		if (v<=-1) {
			// this case should not happen; but might happen due to floating number operations
			// y and x are antipodal on their great circle
			// the corresponding log-map will be infinite
			// approximate y with a point closer to x
			v = -0.95;
		}
		theta = Math.acos(v);
		costheta=v;
		sintheta = Math.sqrt(1-v*v);
		for (int i=0; i<y.length; i++)
			z[i] = theta*(y[i]-x[i]*costheta)/sintheta;
		return z;
	}
	
	protected double[] MapEXP(double[] x, double[] z) {
		// inverse of MapLOG
		// project z on the tangent space of point x back to point y on the sphere
		
		/* the following property holds
		 * norm(z) = geodesic(x,y) = r*theta(x,y) ; must be at most pi * r
		 * therefore, theta = norm(z)/r
		*/
		double theta = MyMatrix.norm(z)/radius;
		double costheta, sintheta;

		if (theta<0.01) {
			// z is VERY close to the origin of the log map
			// simply return origin 
			return Arrays.copyOf(x, x.length);
		}
		
		if (theta > Math.PI) {
			// this should not happen, but might because of floating number operations
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
	
	
	
	static MDS_Sphere bestMDS(double[][] F1, int m_min, int m_max, double k_min, double k_max, double k_step) {
		// find the best dimension, best curvature
		MDS_Sphere best_mds = null;
		double min = Double.MAX_VALUE;
		for (int m1=m_min; m1 <= m_max; m1++) {
			double k1 = k_min;
			while(k1 <= k_max) {
				if (k1 > 0 && k1 <  Math.PI*Math.PI) {
					MDS_Sphere mds = new MDS_Sphere(m1, k1);
					mds.loadDissimMatrix(F1);
					if (mds.embed()) {
						if (mds.embedding_distortion < min) {
							min = mds.embedding_distortion;
							best_mds = mds;
						}
					}	
				}
				
				k1 += k_step;
			}	
		}
		return best_mds;
	}
	
}
