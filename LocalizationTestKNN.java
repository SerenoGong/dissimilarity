
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;

public class LocalizationTestKNN extends Localization {
	
	boolean[][] W; // W[i][j]=true iff the entry dissimilarity F[i][j] is known
	double knn_graph_location_err; // esimate dissim[i][j] based on length of shortest path between i and j
	
	
	LocalizationTestKNN(String location_filename, String dissim_filename, String W_filename, String L_filename) {
		loadLocationMatrix(location_filename);
		loadDissimilarityMatrix(dissim_filename);	
		loadMatrixW(W_filename);
		loadLocationTraining(L_filename);
		kNN_ShortestPath();
	}
	private void loadMatrixW(String filename) {
		int[][] W1 = MyMatrix.loadFromFile_Int(filename);
		W = new boolean[n][n];
		for (int i=0; i < W1.length; i++)
			for (int j=0; j<W1[0].length; j++)
				W[i][j] = (W1[i][j] == 1)? true: false;
	}
	
	
	
	void kNN_ShortestPath() {
		// compare to the method using shortest-path length to estimate unknown dissimilarity
		double[][] adjacency_matrix = new double[n][n];
		for (int i=0; i<n; i++)
			for (int j=0; j<i; j++) {
				adjacency_matrix[i][j] = (W[i][j])? F[i][j]: 0;
				adjacency_matrix[j][i] = adjacency_matrix[i][j];
			}
		FloydWarshall floydwarshall = new FloydWarshall(n);
        floydwarshall.floydwarshall(adjacency_matrix);
        knn_graph_location_err =  knnLocalization(floydwarshall.distancematrix);
	}
	
	
	
	double kNN_Sphere(SphereMDS mds) {
		// test kNN localization
		double[][] dissim = new double[n][n];
		for (int i=0; i<n; i++)
			for (int j=0; j<i; j++) {
				dissim[i][j] = (W[i][j])? F[i][j]:mds.geodesic_dist[i][j];
				dissim[j][i] = dissim[i][j];
			}
		return knnLocalization(dissim);
	}
	
	
	double kNN_Euclidean(MDS mds) {
		// test kNN localization
		double[][] dist = mds.distance(mds.X);
		double[][] dissim = new double[n][n];
		for (int i=0; i<n; i++)
			for (int j=0; j<i; j++) {
				dissim[i][j] = (W[i][j])? F[i][j]:dist[i][j];
				dissim[j][i] = dissim[i][j];
			}
		
		return knnLocalization(dissim);
	}
	
	void test1(String filename) {
		// for each m, print all results for all values o k
		
		int min_m=2;
		int max_m=10;
		int num_iterations = 20;
	
		MDS mds_e = new MDS(F);
		
		try {
			PrintWriter pw = new PrintWriter(filename);
			pw.println("m, best_k, train_err, test_err, knn_err, knn_euclidean, knn_graph");
			for (int m=min_m; m<=max_m; m++) {
				SphereMDSPartial mds1 = SphereMDSPartial.bestMDS_IncompleteDissim(F, W, m, m, num_iterations);
				System.out.println("bestMDS_IncompleteDissim() done, best m=" + m + 
							", best k=" + mds1.k + ", train_err=" + mds1.train_error + ", test_err=" + mds1.test_error);
				
				double knn_err = kNN_Sphere(mds1);
				System.out.println("knn_Sphere() done: knn_err=" + knn_err);
				
				
				// euclidean MDS error
				mds_e.setDimension(m);
				mds_e.embed_IncompleteDissim(W, num_iterations);
				double knn_err_euclidean = kNN_Euclidean(mds_e);
				System.out.println("kNN_Euclidean() done: knn_err=" + knn_err_euclidean);

				pw.println(m + "," + mds1.k + "," + mds1.train_error + "," + mds1.test_error + "," 
							+ knn_err + "," + knn_err_euclidean + "," + knn_graph_location_err);
			}
			pw.close();
		}
		catch (IOException e) {
			System.err.println("file error");
			System.exit(-1);
		}
	}
		
		
	void test2() {
		
		int min_m=3;
		int max_m=4;
		int num_iterations = 30;
		
		
		System.out.println("Euclidean Embedding: ");
		MDS mds = new MDS(F);
		
		for (int m=min_m; m <= max_m; m++) {
			mds.setDimension(m);
			mds.embed_IncompleteDissim(W, num_iterations);		
			double knn_err_euclidean = kNN_Euclidean(mds);
			System.out.println("kNN done, m=" + m + ", knn_err=" + knn_err_euclidean);
		}
		
		
	}
	
	public static void main(String[] args) {
		
		// to generate input files, un-comment the 2 lines below
		//CaseStudy study = new CaseStudy("input_data");
		//study.generateFiles();
		
		//INPUT
		
		int n=300; // 300, 500, 1000
		int d=3; // 3
		String working_dir = "input_data";
		String surface = "sphere";
		String dissim = "geodesic_linear";
		
		int known_dissim_prob=20; // 20% of matrix F is known
		int location_train_prob = 50; // 50% of points have known locations
		
		String location_filename = working_dir + "/" + surface  + "/D"+d + "N"+n;
		String dissim_filename = location_filename + "_" + dissim;
		String W_filename = working_dir + "/W_N" + n + "p" + known_dissim_prob;
		String L_filename = working_dir + "/L_N" + n + "p" + location_train_prob;
		
		
		System.out.println("location file: " + location_filename);
		System.out.println("dissim file: " + dissim_filename);
		System.out.println("W file: " + W_filename);
		System.out.println("L file: " + L_filename);
		
		// OUPUT
		LocalizationTestKNN test = new LocalizationTestKNN(location_filename, dissim_filename, W_filename, L_filename);
//		test.test1(dissim_filename + "_embed_sphere.csv");
		test.test2();
	}
}