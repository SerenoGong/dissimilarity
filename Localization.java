import java.util.ArrayList;
import Jama.Matrix;

public class Localization {
	
	int n; // number of nodes (points)
	int d; // location dimension
	
	double[][] F; // input dissimilarity matrix
	
	double[][] Ytruth; // ground-truth location of point
	double[][] euclidean_truth; // ground-truth euclidean distance between points
	
	// only nodes in location_trainNodes have location known
	ArrayList<Integer> location_trainNodes, location_testNodes;
	
	void loadDissimilarityMatrix(String filename) {
		F = MyMatrix.loadFromFile(filename);
		F = MyMatrix.normalize_01(F);
//		System.out.println("Dissimilarity Matrix F has rank " + new Matrix(F).rank());
	}

	void loadLocationMatrix(String filename) {
		Ytruth = MyMatrix.loadFromFile(filename);
		n = Ytruth.length;
		d = Ytruth[0].length;
		
		euclidean_truth = new double[n][n];
		for (int i=0; i<n; i++) {
			for (int j=0; j<i; j++) {
				euclidean_truth[i][j] = Misc.euclideanDist(Ytruth[i], Ytruth[j]);
				euclidean_truth[j][i] = euclidean_truth[i][j];
			}
		}
	}
	
	void loadLocationTraining(String filename) {
		int[][] LL = MyMatrix.loadFromFile_Int(filename);
		int[] L = LL[0]; // the file should consist of only 1 line
		
		// set location training data
		location_trainNodes = new ArrayList<Integer>();
		location_testNodes = new ArrayList<Integer>();
		for (int i: L) location_trainNodes.add(i);
		for (int i=0; i<n; i++) 
			if (!location_trainNodes.contains(i))
				location_testNodes.add(i);
		
	}
	
	
	void localizationMR(double[][] X, double[][] W) {
		boolean[] label = new boolean[n];
		for (int i : location_testNodes) label[i] = true;
		LocalizationMR mr  = new LocalizationMR(X, Ytruth, label);
		mr.run(1, 0.1, 1, W);
		System.out.println("..........location error: " + mr.locationError());
	}
	
	
	double knnLocalization(double[][] dissim) {
		// return average location error of location_testNodes
		double error = 0;
		for (int i : location_testNodes) {
			// find Nearest neighbor
			int min_j = -1;
			double min = Double.MAX_VALUE;
			for (int j : location_trainNodes) {
				if (dissim[i][j] < min) {
					min = dissim[i][j];
					min_j = j;
				}
			}
			error += Misc.euclideanDist(Ytruth[i], Ytruth[min_j]);
		}
		return error/(double) location_testNodes.size();
	}
	
}