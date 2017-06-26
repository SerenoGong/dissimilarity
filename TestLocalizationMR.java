import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Random;

import Jama.Matrix;

public class TestLocalizationMR extends Localization {
	
	// only nodes in dissimilarity_trainNodes have dissimilarity known with every node
	ArrayList<Integer> dissimilarity_trainNodes, dissimilarity_testNodes;
	double[][] Ftrain; // dissimilarity matrix involving train nodes only

	// for convenience
	double curvature_k;
		
	TestLocalizationMR(String location_filename, String dissim_filename) {
		loadLocationMatrix(location_filename);
		loadDissimilarityMatrix(dissim_filename);	
	}
	
	void setupDissimilarityTraining(double dissimilarity_prob) {
		// set training data for dissimilarity training
		// if the set of dissimilarity training node = V
		// that means F[i][j] is known for every i,j in V
		
		dissimilarity_trainNodes = new ArrayList<Integer>();
		dissimilarity_testNodes = new ArrayList<Integer>();
		for (int i=0; i< n; i++) {
			if (Math.random() < dissimilarity_prob) dissimilarity_trainNodes.add(i);
			else dissimilarity_testNodes.add(i);
		}
		System.out.println("dissimilarity_trainSize: " + dissimilarity_trainNodes.size() + "/" + n);
		
		// compute Ftrain
		Ftrain = new double[dissimilarity_trainNodes.size()][dissimilarity_trainNodes.size()];
		for (int i=0; i<dissimilarity_trainNodes.size(); i++) {
			for (int j=0; j<i; j++) {
				Ftrain[i][j] = F[dissimilarity_trainNodes.get(i)][dissimilarity_trainNodes.get(j)];
				Ftrain[j][i] = Ftrain[i][j];
			}
		}
	}
	
	double[][] runSphereEmbedding(int min_m, int max_m) {
		// now use Ftrain to compute the geometry of the dissimilarity space
		// need to find the best k for a given m	
		MDS_Sphere bestMDS = MDS_Sphere.bestMDS(Ftrain, min_m, max_m, 0, Math.PI*Math.PI, 1);
		if (bestMDS == null) {
			System.err.println("runSphereEmbedding(): Training failed");
			System.exit(-1);
		}
		int m = bestMDS.m+1; // always 1 more than the manifold_dimension
		System.out.println("SPHERICAL best_dimension between " + min_m + " and " + max_m + " is " + m);		
		System.out.println("..........best_curvature k=" + bestMDS.curvature + ", distort= " + bestMDS.embedding_distortion);
		
		curvature_k = bestMDS.curvature;
		
		double[][] Xtrain = bestMDS.X;
		
		// now, bestk is the best curvature and Xtrain is the projection locations of trainNodes
		// next, compute the projection locations of testNodes
		// given a test point, based on its dissimilarity with a train point, we can find its location on the sphere
	
		double[][] X = new double[n][m];
		for (int i=0; i< dissimilarity_trainNodes.size(); i++) 
			X[dissimilarity_trainNodes.get(i)] = Xtrain[i];
		
		
		double[] random_point = bestMDS.randomPoint();
				
		for (int i: dissimilarity_testNodes) {
			double[] f = new double[dissimilarity_trainNodes.size()];
			for (int j=0; j<dissimilarity_trainNodes.size(); j++) f[j] = F[i][dissimilarity_trainNodes.get(j)];
			
			// find 3 nearest trainNodes based on F
			int knn = 3;
			int[] indices = Misc.getKMIN(f, knn);
			double[][] y = new double[knn][m];
			double[] distances = new double[knn];
			for (int j=0; j<knn; j++) {
				y[j] = Xtrain[indices[j]]; 
				distances[j] = f[indices[j]];
			}
			X[i] = bestMDS.multilateration(random_point, y, distances, 20);
		}
		
		// compute error
		double[][] dist = bestMDS.geodesic(X);
		//System.out.println("..........testing_distort= " + MyMatrix.normFrobenius(dist, F)/Math.sqrt(n*(n-1)));
		//System.out.println("..........dissimilarity matrix (estimate) has rank " + new Matrix(dist).rank());
		
		double err = 0;
		for (int i: dissimilarity_testNodes) {
			for (int j: dissimilarity_testNodes)
				if (j<i) err += Math.abs(dist[i][j] - F[i][j])/F[i][j];
		}
		double s = dissimilarity_testNodes.size();
		err = err / (s*(s-1)/2);
		System.out.println("..........dissimilarity between testing nodes = " + 100*err + "% of ground-truth");	
		return X;
	}
	double[][] runEuclideanEmbedding(int min_m, int max_m) {
		MDS bestMDS = MDS.bestMDS(Ftrain, min_m, max_m);
		if (bestMDS == null) {
			System.err.println("runEuclideanEmbedding(): Training failed");
			System.exit(-1);
		}
		int m = bestMDS.m; 
		System.out.println("EUCLIDEAN best_dimension between " + min_m + " and " + max_m + " is " + m);		
		System.out.println("..........distort= " + bestMDS.train_err);
		
		double[][] Xtrain = bestMDS.X;
		
		// next, compute the projection locations of testNodes
		// given a test point, based on its dissimilarity with a train point, we can find its location on the sphere
	
		double[][] X = new double[n][m];
		for (int i=0; i< dissimilarity_trainNodes.size(); i++) 
			X[dissimilarity_trainNodes.get(i)] = Xtrain[i];
		
		
		for (int i: dissimilarity_testNodes) {
			double[] f = new double[dissimilarity_trainNodes.size()];
			for (int j=0; j<dissimilarity_trainNodes.size(); j++) f[j] = F[i][dissimilarity_trainNodes.get(j)];
			
			// find 3 nearest trainNodes based on F
			int knn = 3;
			int[] indices = Misc.getKMIN(f, knn);
			double[][] y = new double[knn][m];
			double[] distances = new double[knn];
			for (int j=0; j<knn; j++) {
				y[j] = Xtrain[indices[j]]; 
				distances[j] = f[indices[j]];
			}
			X[i] = bestMDS.multilateration(new double[m], y, distances);
		}
		
		// compute error
		double[][] dist = bestMDS.distance(X);
		//System.out.println("..........testing_distort= " + MyMatrix.normFrobenius(dist, F)/Math.sqrt(n*(n-1)));
		//System.out.println("..........dissimilarity matrix (estimate) has rank " + new Matrix(dist).rank());
		
		double err = 0;
		for (int i: dissimilarity_testNodes) {
			for (int j: dissimilarity_testNodes)
				if (j<i) err += Math.abs(dist[i][j] - F[i][j])/F[i][j];
		}
		double s = dissimilarity_testNodes.size();
		err = err / (s*(s-1)/2);
		System.out.println("..........dissimilarity between testing nodes = " + 100*err + "% of ground-truth");	
		return X;
	}

	
	void runTestMR() {
		// compare localization results
		
		
		int min_m=2;
		int max_m=10;
		
		
		double[][] W = new double[n][n];
		double gaussianCoeff = 1;
		
		// Euclidean embedding 
		double[][] X = runEuclideanEmbedding(min_m, max_m);
		for (int i=0; i<n; i++)
			for (int j=0; j<i; j++) {
				double dist = Misc.euclideanDist(X[i], X[j]);
				W[i][j] = Math.exp(dist*dist*(-1)/2/gaussianCoeff/gaussianCoeff);
				W[j][i] = W[i][j];
			}
		localizationMR(X, W);
		
		// Spherical embedding
		X = runSphereEmbedding(min_m, max_m);
		W = new double[n][n];
		for (int i=0; i<n; i++)
			for (int j=0; j<i; j++) {
				double dist = Misc.euclideanDist(X[i], X[j]);
				W[i][j] = Math.exp(dist*dist*(-1)/2/gaussianCoeff/gaussianCoeff);
				W[j][i] = W[i][j];
			}
		localizationMR(X, W);
		
		
		System.out.println("use F in place of W:");
		W = new double[n][n];
		for (int i: dissimilarity_trainNodes) 
			for (int j=0; j<n; j++) W[i][j] = (i==j)? 0: 1-F[i][j];
		
		MDS_Sphere sphere = new MDS_Sphere(X[0].length, curvature_k);
		for (int i: dissimilarity_testNodes) {
			for (int j: dissimilarity_trainNodes) W[i][j] = 1-F[i][j];
			for (int j: dissimilarity_testNodes) {
				W[i][j] = (i==j)? 0: 1-sphere.geodesic(X[i], X[j]);
			}
		}
		localizationMR(X, W);
		
		System.out.println("use F in place of W: try 2 (SEEMS BETTER THAN TRY 1)");
		W = new double[n][n];
		for (int i: dissimilarity_trainNodes) 
			for (int j=0; j<n; j++) W[i][j] = (i==j)? 0: 1-F[i][j];
		
		for (int i: dissimilarity_testNodes) {
			for (int j: dissimilarity_trainNodes) W[i][j] = 1-F[i][j];
			for (int j: dissimilarity_testNodes) {
				W[i][j] = (i==j)? 0: 0;//1-RiemannianMDS.geodesic(X[i], X[j], curvature_k);
			}
		}
		localizationMR(X, W);
		
		// localization using Dissimilarity Space
		// use original dissimilarity train + estimated dissimilarity values
		double[][] Z = new double[n][n];
		for (int i: dissimilarity_trainNodes) 
			for (int j=0; j<n; j++) Z[i][j] = F[i][j];
		
		for (int i: dissimilarity_testNodes) {
			for (int j: dissimilarity_trainNodes) Z[i][j] = F[i][j];
			for (int j: dissimilarity_testNodes) {
				Z[i][j] = (i==j)? 0: sphere.geodesic(X[i], X[j]);
			}
		}
		
		W = new double[n][n];
		for (int i=0; i<n; i++)
			for (int j=0; j<i; j++) {
				double dist = Misc.euclideanDist(Z[i], Z[j]);
				W[i][j] = Math.exp(dist*dist*(-1)/2/gaussianCoeff/gaussianCoeff);
				W[j][i] = W[i][j];
			}
		
		
		System.out.print(".........." + n+"-DISSIMILARITY_SPACE ");
		localizationMR(Z, W);
		
		
		
		// using original dissimilarity train 
		int m=dissimilarity_trainNodes.size(); 
		Z = new double[n][m];
		for (int i=0;  i<n; i++) {
			for (int j=0; j<m; j++) {
				Z[i][j] = F[i][dissimilarity_trainNodes.get(j)];
			}
		}
		W = new double[n][n];
		for (int i=0; i<n; i++)
			for (int j=0; j<i; j++) {
				double dist = Misc.euclideanDist(Z[i], Z[j]);
				W[i][j] = Math.exp(dist*dist*(-1)/2/gaussianCoeff/gaussianCoeff);
				W[j][i] = W[i][j];
			}
		
		System.out.println("NO_EMBED: " + m+"-DISSIMILARITY_SPACE");
		localizationMR(Z, W);
		
		// using original dissimilarity train 
		m=X[0].length;
		Z = new double[n][m];
		for (int i=0;  i<n; i++) {
			for (int j=0; j<m; j++) {
				Z[i][j] = F[i][dissimilarity_trainNodes.get(j)];
			}
		}
		W = new double[n][n];
		for (int i=0; i<n; i++)
			for (int j=0; j<i; j++) {
				double dist = Misc.euclideanDist(Z[i], Z[j]);
				W[i][j] = Math.exp(dist*dist*(-1)/2/gaussianCoeff/gaussianCoeff);
				W[j][i] = W[i][j];
			}
		System.out.println(".........." + m+"-DISSIMILARITY_SPACE");
		localizationMR(Z, W);

			
	}
	
	
	public static void main(String[] args) {
		// to generate input files, un-comment the 2 lines below
		//CaseStudy study = new CaseStudy("input_data");
		//study.generateFiles();
				
		//INPUT
				
		int n=100; // 300, 500, 1000
		int d=3; // 3
		String working_dir = "input_data";
		String surface = "euclidean";
		String dissim = "euclidean_poly";
		
		String location_filename = working_dir + "/" + surface  + "/D"+d + "N"+n;
		String dissim_filename = location_filename + "_" + dissim;
		
		System.out.println("location file: " + location_filename);
		System.out.println("dissim file: " + dissim_filename);
		
		TestLocalizationMR p = new TestLocalizationMR(location_filename, dissim_filename);
		p.setupDissimilarityTraining(.2);
		p.runTestMR();
					
	}
}