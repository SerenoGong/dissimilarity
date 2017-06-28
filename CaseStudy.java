import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;


public class CaseStudy {
	String working_dir;
	
	static String[] SURFACE = {"euclidean", "sphere"};
	static String[] DISTANCE = {"euclidean", "geodesic"};
	static String[] DISSIM = {"linear", "poly", "log", "exp"};
	
	static int[] NUMPOINTS = {100, 500};
	static int[] DIMENSION = {2, 3};
	
	CaseStudy(String dir) {
		working_dir = dir;
	}
	void generateFiles() {
		new File(working_dir).mkdirs();
		Random rand = new Random();
		for (int i1=0; i1 < SURFACE.length; i1++) {
			// create directorry
			String surface = SURFACE[i1];
			new File(working_dir + "/" + surface).mkdirs();	
			for (int n1=0; n1 < NUMPOINTS.length; n1++) {
				int n = NUMPOINTS[n1];
				for (int d1=0; d1 < DIMENSION.length; d1++) {
					int d = DIMENSION[d1];
					double[][] locations = new double[n][d];
					
					switch (i1) {
					case 0:	
						// euclidean surface
						locations = new double[n][d];
						for (int i=0; i<n; i++)
							for (int j=0; j<d; j++)
								locations[i][j] = rand.nextDouble();
						break;
					case 1:
						// spherical surface
						MDS_Sphere mds = new MDS_Sphere(d, 1);
						locations = mds.randomPoint(n);
						break;
					}
					MyMatrix.saveAs(locations, working_dir + "/" + filename_location(n, d, surface));
					
					for (int i2=0; i2 < DISTANCE.length; i2++) {
						String distance = DISTANCE[i2];
						double[][] dist = null;
						switch (i2) {
						case 0:	
							// euclidean distance
							dist = euclideanDissim(locations);
							break;
						case 1:
							// geodesic distance (applied only for non-euclidean surface)
							if (surface.compareTo("euclidean") != 0) 
								dist = geodesicDissim(locations);
							else 
								dist = null;
							break;
						}
						
						for (int i3=0; i3 < DISSIM.length; i3++) {
							if (dist == null) continue;
							String dissim = DISSIM[i3];
							double[][] F = generateDissim(DISSIM[i3], dist);
							MyMatrix.saveAs(F, working_dir + "/" + filename_dissim(n, d, surface, distance, dissim));
						}
					}
				}
			}
			
		}
		
	}
	
	double[][] euclideanDissim(double[][] locations) {
		// compute euclidean distances between points
		int n = locations.length;
		double[][] dist = new double[n][n];
		
		for (int i=0; i<n; i++) {
			for (int j=0; j<i; j++) {
				double[] x = locations[i];
				double[] y = locations[j];
	
				dist[i][j] = Misc.euclideanDist(x, y);
				dist[j][i] = dist[i][j];
			}
		}
		return dist;
	}
		
	
	double[][] geodesicDissim(double[][] locations) {
		// compute geodesic distances between points on a unit sphere
		MDS_Sphere mds = new MDS_Sphere(locations[0].length, 1);
		return mds.geodesic(locations);
	}
	
	
	private double[][] generateDissim(String type, double[][] dist) {
		// generate dissimilarity matrix as a function of ground-truth distance
		// normalize it such that maximum possible value is  1
		int n = dist.length;
		double[][] F = new double[n][n];
		for (int i=0; i< n; i++) {
			for (int j=0; j < i; j++) {
				double e = dist[i][j];
				if (type.compareTo("linear")==0) {
					// linear
					F[i][j] = e;
				}
				if (type.compareTo("poly")==0) {
					// linear
					F[i][j] = e*e;
				}
				if (type.compareTo("log")==0) {
					// linear
					F[i][j] = Math.log(1+e);  //to make F[i][j]>=0 and F[i][j]=0 if e=0
				}
				if (type.compareTo("exp")==0) {
					// linear
					F[i][j] = Math.exp(e)-1; // to make F[i][j]=0 if e=0
				}
				F[j][i] = F[i][j];
			}
		}
		// normalize it such that maximum possible value is  1
		return MyMatrix.normalize_01(F);
	}
	
	void generate_W_Files() {
		for (int n : NUMPOINTS) {
			for (int p=10; p<=100; p+=10) {
				int[][] W = null;
				while(true) {
					// will break if W forms a connected graph
					// else, continue trying to find a good W
					W = MyMatrix.random_01Matrix(n, (double) p/100.0);
					if (Misc.isConnectedGraph(W)) break;
				}
				String W_filename = working_dir + "/" + filename_W(n, p);
				MyMatrix.saveAs(W, W_filename);
			}
		}
	}
	
	void generate_Location_Train_Files() {
		Random rand = new Random();
		for (int n : NUMPOINTS) {
			for (int p=10; p<=100; p+=10) {
				String filename = working_dir + "/" + filename_L(n, p);
				double location_prob=  (double) p / 100.00;
				try {
					PrintWriter pw = new PrintWriter(filename);
					boolean first=true;
					for (int i=0; i<n; i++) 
						if (rand.nextDouble() < location_prob) {
							if (first) {
								pw.print(i);
								first = false;
							}
							else pw.print(","+i);
						}
					pw.close();
				}
				catch (IOException e) {
					System.err.println("generate_Location_Train_Files(): file error" + filename);
					System.exit(-1);
				}
			}
		}
	}
	
	static String filename_location(int n, int d, String surface) {
		return surface + "/D"+d + "N"+n;
	}
	static String filename_dissim(int n, int d, String surface, String distance, String dissim) {
		return filename_location(n,d,surface) + "_" + distance + "_" + dissim;
	}
	static String filename_W(int n, int p) {
		return "W"+p +"N" + n;
	}
	static String filename_L(int n, int p) {
		return "L"+p +"N" + n;
	}
	
	void generateAllFiles() {
		generateFiles();
		generate_W_Files();
		generate_Location_Train_Files();
	}
	
	public static void main(String[] args) {
		
		CaseStudy cs = new CaseStudy("input_synthetic");
		cs.generateAllFiles();
	}

}
