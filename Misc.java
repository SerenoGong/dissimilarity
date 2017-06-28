
import java.util.ArrayList;
import java.util.Vector;

public class Misc {
	   

	public static double euclideanDist(double[] a, double [] b) {
		assert (a.length == b.length) : "euclideanDist(): different array lengths!";
		double ret = 0;
		for (int i=0; i<a.length; i++) {
			ret += (a[i]-b[i]) * (a[i]-b[i]);
		}
		return Math.sqrt(ret);
	}
	
	public static double euclideanDist(double x1, double y1, double x2, double y2) {
		return Math.sqrt((x1-x2) * (x1-x2) + (y1-y2) * (y1-y2));
	}
	
	public static double manhattanDist(double[] a, double [] b) {
		assert (a.length == b.length) : "manhattanDist(): different array lengths!";
		double ret = 0;
		int n= a.length;
		for (int i=0; i<n; i++) {
			ret += Math.abs(a[i]-b[i]);
		}
		return ret;
	}
	
	public static double[] vector2Array(Vector<Double> v) {
		double[] arr = new double[v.size()] ;
		int i = 0;
		for (Double r: v) arr[i++] = r;
		return arr;
	}
	
	public static double euclideanDist(Vector<Double> v1, Vector<Double> v2) {
		double[] r1 = vector2Array(v1);
		double[] r2 = vector2Array(v2);
		return Misc.euclideanDist(r1, r2);
	}
	public static double manhattanDist(Vector<Double> v1, Vector<Double> v2) {
		double[] r1 = vector2Array(v1);
		double[] r2 = vector2Array(v2);
		return Misc.manhattanDist(r1, r2);
	}

	private static int getNN(ArrayList<Double> list, double value) {
		// find the nearest neighbor of p
		// return index in pList of this neighbor
		
		double min_diff = Double.MAX_VALUE;
		int nn = 0;
		for (int i=0; i<list.size(); i++) {
			double diff = Math.abs(list.get(i)-value);
			if (diff < min_diff) {
				min_diff = diff;
				nn = i;
			}
		}
		return nn;
	}
	private static int getMIN(ArrayList<Double> list) {
		// find the minimum number
		// return index in pList of this neighbor	
		double min = Double.MAX_VALUE;
		int nn = 0;
		for (int i=0; i<list.size(); i++) {
			double d = list.get(i);
			if (d < min) {
				min = d;
				nn = i;
			}
		}
		return nn;
	}
	public static int[] getKMIN(double[] list, int k) {
		int[] indices = new int[k];
		ArrayList<Double> list1 = new ArrayList<Double>();
		for (int i=0; i< list.length; i++) list1.add(list[i]);
			
		for (int i=0; i<k; i++) {
			indices[i] = getMIN(list1);
			list1.remove(indices[i]);
		}
		return indices;
	}
	
	public static int[] getKNN(double[] list, double value, int k) {
		int[] indices = new int[k];
		ArrayList<Double> list1 = new ArrayList<Double>();
		for (int i=0; i< list.length; i++) list1.add(list[i]);
			
		for (int i=0; i<k; i++) {
			indices[i] = getNN(list1, value);
			list1.remove(indices[i]);
		}
		return indices;
	}

	static double acosh(double x) {
		return Math.log(x + Math.sqrt(x*x - 1.0));
	}

	
	static boolean isConnectedGraph(int[][] adjacency_matrix) {
		// return TRUE if the graph formed by adjacency_matrix
		int n = adjacency_matrix.length;
		boolean[] visited = new boolean[n]; // by default, all FALSE
		visited[0] = true;
		
		int num_visited=1;
		while(true) {
			boolean stop_flag = true;
			for (int i=0; i<n; i++) {
				if (!visited[i]) continue;
				for (int j = 0; j < adjacency_matrix[i].length; j++) 
			        if (adjacency_matrix[i][j]==1 && visited[j] == false) {
			        	visited[j] =true;
			        	num_visited++;
			        	stop_flag = false;
			        }
			}
			if (stop_flag) break;
		}
		if (num_visited == n) return true;
		else return false;
	}
}

