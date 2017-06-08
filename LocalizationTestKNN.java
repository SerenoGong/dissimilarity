
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;

public class LocalizationTestKNN extends Localization {
	
	boolean[][] W; // W[i][j]=true iff the entry dissimilarity F[i][j] is known
	double knn_graph_location_err; // esimate dissim[i][j] based on length of shortest path between i and j
	
	static int min_m;
	static int max_m;
	static int num_iterations;
	
	LocalizationTestKNN(String location_filename, String dissim_filename) {
		loadLocationMatrix(location_filename);
		loadDissimilarityMatrix(dissim_filename);	
	}
	
	private void loadMatrixW(String filename) {
		int[][] W1 = MyMatrix.loadFromFile_Int(filename);
		W = new boolean[n][n];
		for (int i=0; i < W1.length; i++)
			for (int j=0; j<W1[0].length; j++)
				W[i][j] = (W1[i][j] == 1)? true: false;
	}
	
	void saveEmbedData(String casefilename) {
		for (int m=min_m; m <= max_m; m++) {
			System.out.println("saving files " + casefilename + "_m"+m+".*");
			SphereMDSPartial s_mds = SphereMDSPartial.bestMDS_IncompleteDissim(F, W, m, m, num_iterations);
			MyMatrix.saveAs(s_mds.X, casefilename + "_m"+m+"_sMDS.X");
			MyMatrix.saveAs(s_mds.geodesic_dist, casefilename + "_m"+m+"_sMDS.dist");
			
			// euclidean embedding
			MDS e_mds = new MDS(F);
			e_mds.setDimension(m);
			e_mds.embed_IncompleteDissim(W, num_iterations);
			double[][] dist = e_mds.distance(e_mds.X);
			MyMatrix.saveAs(e_mds.X, casefilename + "_m"+m+"_MDS.X");
			MyMatrix.saveAs(dist, casefilename + "_m"+m+"_MDS.dist");
		}		
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
	
	
	
	
	double knn(double[][] dist) {
		// test kNN localization
		double[][] dissim = new double[n][n];
		for (int i=0; i<n; i++)
			for (int j=0; j<i; j++) {
				dissim[i][j] = (W[i][j])? F[i][j]:dist[i][j];
				dissim[j][i] = dissim[i][j];
			}
		
		return knnLocalization(dissim);
	}
	
	double[] errors(double[][] truth, double[][] est, boolean[][] W) {
		double train_err = 0;
		double test_err = 0;
		int train_count=0;
		int test_count=0;
		
		for (int i=0; i<n; i++) {
			for (int j=0; j<i; j++) {
				double diff = est[i][j]-truth[i][j];
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
		train_err = Math.sqrt(train_err) / (double) train_count;
		test_err = Math.sqrt(train_err) / (double) test_count;
		double[] err= {train_err, test_err};
		return err;
	}
	
	String bestMDS(String casefilename, String whichMDS) {		
		double best_train_err = Double.MAX_VALUE;
		double best_test_err = Double.MAX_VALUE;
		int best_m = 2;
			
		for (int m=min_m; m<=max_m; m++) {
			double[][] dist=null;
			if (whichMDS.compareTo("sphere")==0) dist = MyMatrix.loadFromFile(casefilename + "_m"+m+"_sMDS.dist");
			if (whichMDS.compareTo("euclidean")==0) dist = MyMatrix.loadFromFile(casefilename + "_m"+m+"_MDS.dist");
			double[] err  = errors(F, dist, W);
			if (best_train_err < err[0]) {
				best_train_err = err[0];
				best_test_err = err[1];
				best_m = m;
			}
		}
		return best_m + "," + best_train_err + "," + best_test_err;
	}
	
	void testKNN(String casefilename, String result_filename) {		
		try {
			
			PrintWriter pw = new PrintWriter(result_filename);
			pw.println("m, train_err, test_err, knn_err, e_train_err, e_test_err, e_knn_err, g_knn_err");
			
			
			for (int m=min_m; m<=max_m; m++) {
				double[][] dist = MyMatrix.loadFromFile(casefilename + "_m"+m+"_sMDS.dist");
				double[] err  = errors(F, dist, W);
				double knn_err = knn(dist);
				
				pw.print(m + ","  + err[0] + "," + err[1] + "," + knn_err);
				
				
				// euclidean MDS error
				dist = MyMatrix.loadFromFile(casefilename + "_m"+m+"_MDS.dist");
				
				String str = casefilename + "_m"+m+"_MDS.dist";
				if (str.compareTo("input_data_result_i20/sphere/D3N500_euclidean_exp_W80_m3_MDS.dist")==0) {
	//				System.out.println(dist.length +"," + dist[0].length);
	//				System.out.println(F.length +"," + F[0].length);
	//				System.out.println(MyMatrix.toStr(dist));
				}
				err  = errors(F, dist, W);
				knn_err = knn(dist);
				pw.print("," + err[0] + "," + err[1] + "," + knn_err);
				
				pw.println("," + knn_graph_location_err);
				
			}
			pw.close();
		}
		catch (IOException e) {
			System.err.println("file error");
			System.exit(-1);
		}
	}
	
	
	static void runAndSaveEmbedData(String input_dir, String output_dir) {
		new File(output_dir).mkdirs();
		for (int n : CaseStudy.NUMPOINTS) {
			for (int d: CaseStudy.DIMENSION) {
				for (String surface : CaseStudy.SURFACE) {
					new File(output_dir + "/" + surface).mkdirs();
					String location_filename = CaseStudy.filename_location(n, d, surface);
					for (String distance : CaseStudy.DISTANCE) {
						if (surface.compareTo("euclidean")==0 && distance.compareTo("geodesic")==0) {
							// this case does not apply
							continue;
						}
						
						for (String dissim : CaseStudy.DISSIM) {
							String dissim_filename = CaseStudy.filename_dissim(n, d, surface, distance, dissim);
							
							
							LocalizationTestKNN test = new LocalizationTestKNN(
									input_dir + "/" + location_filename, 
									input_dir + "/" + dissim_filename);
							for (int pW = 20; pW <=100; pW += 20) {
								// 20%, 40%, ...of dissim matrix is known
								test.loadMatrixW(input_dir + "/" + CaseStudy.filename_W(n, pW));
								test.saveEmbedData(output_dir + "/" +  dissim_filename + "_W" + pW);		
							}
						}
					}			
				}
			}
		}
	}

	static void reportingKNN(String input_dir, String output_dir) {
		new File(output_dir).mkdirs();
		for (int n : CaseStudy.NUMPOINTS) {
			for (int d: CaseStudy.DIMENSION) {
				for (String surface : CaseStudy.SURFACE) {
					new File(output_dir + "/" + surface).mkdirs();
					String location_filename = CaseStudy.filename_location(n, d, surface);
					for (String distance : CaseStudy.DISTANCE) {
						if (surface.compareTo("euclidean")==0 && distance.compareTo("geodesic")==0) {
							// this case does not apply
							continue;
						}
						
						for (String dissim : CaseStudy.DISSIM) {
							String dissim_filename = CaseStudy.filename_dissim(n, d, surface, distance, dissim);
							System.out.println("reportingKNN(): dissim_filename=" + dissim_filename);
							
							LocalizationTestKNN test = new LocalizationTestKNN(
									input_dir + "/" + location_filename, 
									input_dir + "/" + dissim_filename);
							
							for (int pW = 20; pW <=100; pW += 20) {
								// 20%, 40%, ... of dissim matrix is known
								test.loadMatrixW(input_dir + "/" + CaseStudy.filename_W(n, pW));
							
								for (int pL = 20; pL <= 100; pL += 20) {
									// 20%, 40%, ... of points have known locations
									test.loadLocationTraining(input_dir + "/" + CaseStudy.filename_L(n, pL));
											
									test.kNN_ShortestPath();
									
									String WL_filename = dissim_filename + "_W" + pW + "_L"+pL+"_knn.csv";
									test.testKNN(
											output_dir + "/" +  dissim_filename + "_W" + pW, 
											output_dir + "/" +  WL_filename);			
								}
							}
						}
					}			
				}
			}
		}
	}
	
	
	
	static String knnErr(String WL_filename) {
		// return the knn_error of the best embedding
		double[][] table = MyMatrix.transpose(MyMatrix.loadFromFile(WL_filename, 1));
		
		int h = Misc.getKMIN(table[3], 1)[0];
		int best_m_sphere = (int) table[0][h];
		double knnErr_best_sphere = table[3][h];
		
		h = Misc.getKMIN(table[6], 1)[0];
		int best_m_euclidean = (int) table[0][h];
		double knnErr_best_euclidean = table[6][h];
		
		double knnErr_g = table[7][0];
		
		return best_m_sphere +"," + best_m_euclidean + 
				"," + knnErr_best_sphere + "," + knnErr_best_euclidean + "," + knnErr_g;
	}
	
	
	
	static void reportingKNN_2(String input_dir, String output_dir) {
		// only report the best embedding for each case
		for (int n : CaseStudy.NUMPOINTS) {
			for (int d: CaseStudy.DIMENSION) {
				for (String surface : CaseStudy.SURFACE) {
					String location_filename = CaseStudy.filename_location(n, d, surface);
					
					String save2File1 = output_dir + "/" +  location_filename+".csv";
					try {
						PrintWriter pw1 = new PrintWriter(save2File1);
						pw1.print("distance,dissim,knn_err_s,knn_err_e,knn_err_g,knn_s_e_ratio");
				
						for (String distance : CaseStudy.DISTANCE) {
							if (surface.compareTo("euclidean")==0 && distance.compareTo("geodesic")==0) {
								// this case does not apply
								continue;
							}
							for (String dissim : CaseStudy.DISSIM) {
								String dissim_filename = CaseStudy.filename_dissim(n, d, surface, distance, dissim);
								System.out.println("reportingKNN_2(): dissim_filename=" + dissim_filename);
								
								
								String save2File2 = output_dir + "/" +  dissim_filename+"_WL.csv";
								
								PrintWriter pw2 = new PrintWriter(save2File2);
								pw2.print("w,l,m_s,m_e,knn_err_s,knn_err_e,knn_err_g,knn_s_e_ratio");
								
								double knn_err_s=0;
								double knn_err_e=0;
								double knn_err_g=0;
								int count=0;
								for (int pW = 20; pW <=100; pW += 20) {
									// 20%, 40%, ... of dissim matrix is known
									for (int pL = 20; pL <= 100; pL += 20) {
										// 20%, 40%, ... of points have known locations
										if (pL==100) break;
										String WL_filename = dissim_filename + "_W" + pW + "_L"+pL+"_knn.csv";
										String str = knnErr(output_dir + "/" +  WL_filename);
										
										pw2.println();
										pw2.print(pW + "," + pL + "," + str+
												"," + Double.parseDouble(str.split(",")[2])/Double.parseDouble(str.split(",")[3]));
										
										knn_err_s += Double.parseDouble(str.split(",")[2]);
										knn_err_e += Double.parseDouble(str.split(",")[3]);
										knn_err_g += Double.parseDouble(str.split(",")[4]);
										count++;
									}
								}
								pw2.close();
								
								knn_err_s /= (double) count;
								knn_err_e /= (double) count;
								knn_err_g /= (double) count;
								pw1.println();
								pw1.print(distance + "," + dissim + 
										"," + knn_err_s + "," + knn_err_e + "," + knn_err_g +
										"," + knn_err_s/knn_err_e);
							}
						}	

						pw1.close();
					}
					
					catch (IOException e) {
						System.err.println("reportingKNN_2() file error");
						System.exit(-1);
					}
				}
			}
		}
		
	}
	public static void main(String[] args) {
		
		// to generate input files, un-comment the lines below
		// new CaseStudy("input_data").generateAllFiles();
		
		min_m = 2;
		max_m = 5;
		num_iterations = 20;
				
		if (false) {
			String input_dir = "input_data1";
			String output_dir = input_dir + "_result_i" + num_iterations;
			
			// run embedding and save embedding data into files		
			runAndSaveEmbedData(input_dir, output_dir);
		}		
		else {
			String input_dir = "input_data1";
			String output_dir = input_dir + "_result_i" + num_iterations;
				// reporting results
			//reportingKNN(input_dir, output_dir);
			reportingKNN_2(input_dir, output_dir);
		}
		
		
		
	
	}
}