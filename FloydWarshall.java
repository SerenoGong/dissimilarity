import java.util.Scanner;
 
public class FloydWarshall
{
    double distancematrix[][];
    private int numberofvertices;
    public static final double INFINITY = Double.MAX_VALUE;
 
    public FloydWarshall(int numberofvertices)
    {
        distancematrix = new double[numberofvertices][numberofvertices];
        this.numberofvertices = numberofvertices;
    }
 
    public void floydwarshall(double adjacencymatrix[][])
    {
        for (int source = 0; source < numberofvertices; source++)
        {
            for (int destination = 0; destination < numberofvertices; destination++)
            {
                distancematrix[source][destination] = adjacencymatrix[source][destination];
            }
        }
 
        for (int intermediate = 0; intermediate < numberofvertices; intermediate++)
        {
            for (int source = 0; source < numberofvertices; source++)
            {
                for (int destination = 0; destination < numberofvertices; destination++)
                {
                    if (distancematrix[source][intermediate] + distancematrix[intermediate][destination]
                         < distancematrix[source][destination])
                        distancematrix[source][destination] = distancematrix[source][intermediate] 
                            + distancematrix[intermediate][destination];
                }
            }
        }
    }
 
    public static void main(String... arg)
    {
    	double adjacency_matrix[][];
        int numberofvertices;
 
        Scanner scan = new Scanner(System.in);
        System.out.println("Enter the number of vertices");
        numberofvertices = scan.nextInt();
 
        adjacency_matrix = new double[numberofvertices][numberofvertices];
        System.out.println("Enter the Weighted Matrix for the graph");
        for (int source = 0; source < numberofvertices; source++)
        {
            for (int destination = 0; destination < numberofvertices; destination++)
            {
                adjacency_matrix[source][destination] = scan.nextInt();
                if (source == destination)
                {
                    adjacency_matrix[source][destination] = 0;
                    continue;
                }
                if (adjacency_matrix[source][destination] == 0)
                {
                    adjacency_matrix[source][destination] = INFINITY;
                }
            }
        }
 
        FloydWarshall floydwarshall = new FloydWarshall(numberofvertices);
        floydwarshall.floydwarshall(adjacency_matrix);
        scan.close();
    }
}