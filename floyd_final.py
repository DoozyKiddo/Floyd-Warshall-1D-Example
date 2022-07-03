from mpi4py import MPI
import networkx as nx
import numpy as np
import warnings

warnings.filterwarnings('ignore')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
INFINITY = np.iinfo(np.int32).max

def main():

    data = open('facebook_combined.txt', 'r')
    startTime = MPI.Wtime()
    G = nx.read_edgelist(data, create_using=nx.DiGraph())
    G = G.to_undirected()
    nodes = sorted(G.nodes)
    adj = nx.adjacency_matrix(G, nodelist=nodes)

    adj[adj == 0] = INFINITY
    dd = adj.todense()
    np.fill_diagonal(dd, 0)
    dd = dd.tolist()

    with open('matrix.txt', 'w') as testfile:
        for row in dd:
            testfile.write('   '.join([str(a) for a in row]) + '\n')

    startTime = MPI.Wtime()
    finishedMatrix, finishedRank = run(dd)
    if rank == 0:
        outputTxt = open("output.txt", "w")
        print(finishedMatrix)
        print()
        closeness_centrality = closeCen(dd, nodes)
        print(closeness_centrality)
        outputTxt.write(str(closeness_centrality) + "\n\n")
        finishTime = MPI.Wtime() - startTime
        print("\nRuntime: " + str(finishTime) + " seconds\n")
        outputTxt.write("Runtime: " + str(finishTime) + " seconds\n\n")

        topCentralities = []
        average = sum(closeness_centrality.values()) / len(closeness_centrality)

        for i in range(5):
            maxValue = max(closeness_centrality.values())
            maxKey = max(closeness_centrality, key=closeness_centrality.get)
            print("#" + str(i + 1) + " " + str(maxKey) + " : " + str(maxValue))
            outputTxt.write("#" + str(i + 1) + " " + str(maxKey) + " : " + str(maxValue) + "\n")
            topCentralities.append(closeness_centrality[maxKey])  # Use existing dictionary values to find their centralities
            closeness_centrality.pop(maxKey)

        print("\nAverage closeness centrality of all nodes:", average)
        outputTxt.write("\nAverage closeness centrality of all nodes: " + str(average))

        outputTxt.close()



def getRoot(size, n_psums, k):
    result = size - 1
    for i in range(size):
        if n_psums[i] < k + 1 <= n_psums[i+1]:
            result = i
            break
    return result


def run(graph):
    # n_p is number of rows per processor
    # Graph is assumed to be n x n (square)
    n_p = []
    n_psums = [0]
    n = len(graph)
    remainder = n % size
    for i in range(size):
        temp = int(n / size)
        if remainder > 0:
            temp += 1
            remainder -= 1
        n_p.append(temp)
        n_psums.append(sum(n_p[:i+1]))
    begin = sum(n_p[:rank])
    end = sum(n_p[:rank+1])
    for k in range(n):
        r_proc = getRoot(size, n_psums, k)
        graph[k] = comm.bcast(graph[k], root=r_proc)
        for i in range(begin, end):
            if graph[i][k] == INFINITY:
                continue
            for j in range(n):
                g_ijk = graph[i][k] + graph[k][j]
                if g_ijk < graph[i][j]:
                    graph[i][j] = g_ijk
    for k in range(n):
        r_proc = getRoot(size, n_psums, k)
        graph[k] = comm.bcast(graph[k], root=r_proc)

    return graph, rank


def closeCen(D, nodes):
    n = len(D)
    closeness_centrality = {}
    for r in range(n):
        possible_paths = enumerate(D[r][:])
        shortestPaths = dict(filter(lambda x: x[1] != INFINITY, possible_paths))

        totalShortestPaths = sum(shortestPaths.values())
        if totalShortestPaths > 0.0 and n > 1:
            closeness_centrality[nodes[r]] = (len(shortestPaths) - 1) / totalShortestPaths

    return closeness_centrality


if __name__ == '__main__':
    main()