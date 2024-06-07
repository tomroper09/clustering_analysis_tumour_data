import numpy as np
import matplotlib.pyplot as plt
from validation_scores import silhouettescore

"""
CLUSTERING TECHNIQUES.PY

Contains all three of our clustering algorithms (DBSCAN, K-means and GMM)

VARIABLE DICTIONARY:

K = total number of clusters, k = kth cluster
Data = numpy array of features and data instances
cluster_assignment = 1-d array of which cluster each data point assigns to (k=0 -> K) 
centroids = Middle point of the cluster, also denoted as Mu in the notation for GMM in paper

"""

class dbscan():
          
    """
       DBSCAN - Density-Based clustering, where the clusters are classified as regions of high 
       spatial density. 
              - Amount of clusters unassumed
              - points within eps radius considered density reachable
              - point with 'minpoints' density-reachable points considered core point
              - points without 'minpoints' neighbours but a core point neighbour are considered
              border point
              - non-core, non-border points are considered noise
       
   """ 

    def __init__(self,data,eps,minpoints):
        
       #initialising attributes
       
       self.data = data
       self.eps = eps
       self.minpoints = minpoints
        
    def neighbourhood(self):
        
        """
        function to calculate neighbourhood of each data point
        Outout corresponds to the neighbourhood of each point in the dataset
        Each row of the neighbourhoods array correlates to the same row in the dataset
        """        
        # intialising neighbourhood array
        
        neighbourhoods = [0]*len(self.data)
        
        # calculating neighbourhood for each data point
        
        for point in range(len(self.data)):
            
            # initialising array to hold neighbours of each point
            
            neighbours = []
            
            for row in range(len(self.data)):
            
                # if points within certain distance within each other, neighbours
                
                if np.linalg.norm(self.data[row] - self.data[point]) <= self.eps:
                    
                    neighbours.append(row)
           
            # adding neighbours of each point into correct position
            
            neighbourhoods[point] = neighbours
            
        return neighbourhoods
            
    def dbscan_clustering(self):
        
        """
        Finding the number of clusters and the datapoints within each cluster
        Calculates whetehr a point is a core point, and whether any of its neighbours are
        Continues adding preprocessed core points into new clusters until all points are proessed
        
        """        
        
        # calling neighbourhood function to calculate neighbours of each point
        
        neighbourhoods = dbscan(self.data,self.eps,self.minpoints).neighbourhood()
        
        # initialising clusters
        
        clustering = [0]*len(self.data)
        K = 0

        for current_point in range(len(self.data)):
            
            # if point is unprocessed core point, begin new cluster
            # continue until all points processed
            
            if len(neighbourhoods[current_point]) >= self.minpoints:
                
                # if point is core point, and not part of cluster, create new cluster
                
                if clustering[current_point] == 0:
                
                    K += 1
                    current_cluster = [current_point]
                    
                    # if core pointt in neighbourhood of other core points unprocessed,
                    # add to list and consider part of current cluster
                    
                    while current_cluster:
                    
                        # create new array made of unprocess core points
                        
                        point_to_check = current_cluster.pop()
                        clustering[point_to_check] = K
                        
                        for neighbour in neighbourhoods[point_to_check]:
                            
                            if clustering[neighbour] == 0:
                                
                                # if neighbour is unprocessed core point, add to points_to_check to 
                                # cluster
                                
                                if len(neighbourhoods[neighbour]) >= self.minpoints:
                                    
                                    current_cluster.append(neighbour)
                               
                                else:
                                    
                                # otherwise, border point, just add to cluster
                                
                                    clustering[neighbour] = K
        
        
        # storing clusters as an array of cluster numbers
        
        cluster_assignment = np.vstack(clustering).flatten()
        
        # calculating number of noise points with given eps,minpoints
        
        noise=[]
        
        for row in range(len(cluster_assignment)):
            
            # if point still unprocessed, classified as noise (cluster_assignment == 0)
        
            if cluster_assignment[row] < 1:
                
                # if unprocessed, add to array or noise
            
                noise.append([self.data[row]])
                
        # writing noise as one large numpy array for easier viewing
        
        if len(noise) > 1:
            
            noise=np.vstack(noise)
        
        print('clustering complete. Amount of clusters:',K
              ,' amount of noise values: ',len(noise))
        
        
        return noise,cluster_assignment

    def dbscan_visualisation(self,cluster_assignment,PCA=False):
        
        
        """
        Visualising the cluster results
        If PCA-transformed: uses two larget PCs
        Else: Chooses two random features from the dataset and plots them against
        each other
        Colour codes plot based on which cluster each point is part of
        """
        
        # If PCA-transformed, use two largest PCs
        
        if PCA == True:
            
            # Two largest PCs
            
            chosen_columns = self.data[:,:2]
            
            # Initialise figure
            
            plt.figure(figsize=(10,6))
            
            rcParams['font.weight'] = 'bold'
            
            # plotting two PCs against each other, colour coding using cluster_assignment
            
            plt.scatter(chosen_columns[:,0], chosen_columns[:,1], c = cluster_assignment) 
            plt.colorbar(label = 'Cluster', ticks = range(0,np.max(cluster_assignment+1)))
            
            # creating axis labels 
            
            plt.xlabel('PC1', weight = 'bold')
            plt.ylabel('PC2', weight = 'bold')
            plt.title('DBSCAN Clustering on PC1 and PC2', weight = 'bold')
            
            # updating font size
            plt.rcParams.update({'font.size': 21})
            
        else:
            
            # If not PCA-transformed, pick two random features to plot against each other
           
            chosen_columns = np.random.choice(self.data.shape[1], size = 2, replace = False)
            
            # for axis laelling
            
            chosen_features = np.random.choice(self.data.shape[1], size = 2,replace = False)
            
            
            plt.figure(figsize=(10,6))
            
            
            # plot the two randomly-chosed columns against each other
            # colour code depending on cluster_assignment

            plt.scatter(chosen_columns[:,0], chosen_columns[:,1], c = cluster_assignment) 
            plt.colorbar(label='Cluster', ticks = range(0,np.max(cluster_assignment+1)))
            
            # axis labelling using feature 
            
            plt.xlabel(chosen_features[0])
            plt.ylabel(chosen_features[1])
            
            plt.title('DBSCAN Clustering on Two Features')
        
        
    
            
# Knn Class 
            
class kmeans():

    def __init__(self, data, K):
        """
        Class k-means algorithm, takes inputs 
            1. data - Data Array of features
            2. K- number of partitions/clusters to create in the dataset
        The algorithm will run with 30 random initialisations and pick the best output based on Sil scores
        """

        # Assign selfs
         
        self.data = data
        self.K = K


    def kplusplus_initialisation(self, seed=57):
        """
        Find first centroid randomly from a data point
        Find the rest according to K++ algorithm
        """

        # For reproducibility 

        np.random.seed(seed)

        # Initialise required variables

        N, D = self.data.shape
        centroids = np.empty((self.K, D))

        # First centroid is a random choice 

        first_center_idx = np.random.choice(N)
        centroids[0] = self.data[first_center_idx]

        # Rest are initialised as outlined in K++ paper 
        # Compute the squared distances from each point to the first chosen center

        min_dist = np.sum((self.data - centroids[0])**2, axis=1)

        # Loop through k initialisations, starting on the 2nd centroid
        
        for i in range(1, self.K):

            # Choose a new data point as a new center using a weighted probability distribution

            potential = min_dist.sum()
            weights = min_dist / potential
            new_center_idx = np.random.choice(N, p=weights)
            centroids[i] = self.data[new_center_idx]
            
            # Update the distances to include the minimum distance to all chosen centers

            new_distances = np.sum((self.data - centroids[i])**2, axis=1)
            min_dist = np.minimum(min_dist, new_distances)
            
        return centroids
    

    def create_clusters(self, iter=50):

        """
        K-Means algorithm
        Gives two outputs (centroids = Final centroids of generated clusters, cluster_assignment = Array that classifies each datapoint in X according to its cluster)
        For an initial condition, will loop the k-means algorithm 50 times (iterations) unless convergence is reached
        """

        centroids = self.kplusplus_initialisation()

        # Initialize cluster assignments

        cluster_assignment = np.zeros(self.data.shape[0])

        for _ in range(iter):

            # Create a list of 1-d arrays that describe squared euclidean distances between X data points and each centroid 
            # (each 1-d array corresponds to different centroid)

            distances = [np.sum((self.data - centroid) ** 2, axis=1) for centroid in centroids]

            # then just find the smallest of the two

            new_assignment = np.argmin(distances, axis=0)

            # Check for convergence: if no change in cluster assignment, then stop

            if np.array_equal(new_assignment, cluster_assignment):
                break
            
            # Update cluster assignments

            cluster_assignment = new_assignment

            # Update centroids (this is self explanatory)

            for j in range(self.K):
                centroids[j] = np.mean(self.data[cluster_assignment == j], axis=0)

        return centroids, cluster_assignment
    

    def optimise_clusters(self, num_runs = 30):
        """ 
        This function runs the KNN algorthm 30 times
        Each run corresponds to a different initial condition for the centroids
        It aims to find the optimal initial starting centroids to ensure convergence to the best clustering output
        It skips times when clusters are empty
        """

        # Initialise variables

        best_score = -999
        best_centroids = None
        best_assignment = None
    
        for _ in range(num_runs):
      
            centroids, cluster_assignment = self.create_clusters()
            
            # Calculate the score for the current run

            score = silhouettescore(self.data, cluster_assignment).score()
            
            if score > best_score:
                best_score = score
                best_centroids = centroids
                best_assignment = cluster_assignment
        
        
        return best_centroids, best_assignment




class gmm():

    def __init__(self, data, K, centroids=np.array([])):
        """
        Gaussian Mixture Model algorithm
        Requires three inputs (X = data array, K = Number of clusters, centroids (aka mus in paper, but for consistency in code denoted as centroids))
        If centroids aren't provided will randomly initialise and find the best result
        """

        # Define self
        self.data = data
        self.K = K
        self.centroids = centroids


    def kplusplus_initialisation(self, seed=57):
        """
        Find first centroid randomly from a data point
        Find the rest according to K++ algorithm
        """

        # For reproducibility 

        np.random.seed(seed)

        # Initialise required variables

        N, D = self.data.shape
        centroids = np.empty((self.K, D))

        # First centroid is a random choice 

        first_center_idx = np.random.choice(N)
        centroids[0] = self.data[first_center_idx]

        # Rest are initialised as outlined in K++ paper 
        # Compute the squared distances from each point to the first chosen center

        min_dist = np.sum((self.data - centroids[0])**2, axis=1)

        # Loop through for all different centroids required
        
        for i in range(1, self.K):

            # Choose a new data point as a new center using a weighted probability distribution

            potential = min_dist.sum()
            weights = min_dist / potential
            new_center_idx = np.random.choice(N, p=weights)
            centroids[i] = self.data[new_center_idx]
            
            # Update the distances to include the minimum distance to all chosen centers

            new_distances = np.sum((self.data - centroids[i])**2, axis=1)
            min_dist = np.minimum(min_dist, new_distances)
            
        return centroids
    

    def multivariate_norm(self, mu, sigma):

        # For multivariate normal distribution, equation in paper
        
        d = self.data.shape[1]
        sigma_inv = np.linalg.inv(sigma)
        sigma_det = np.linalg.det(sigma)
        norm_const = 1.0 / (np.power((2*np.pi), float(d)/2) * np.sqrt(sigma_det))
        x_minus_mu = self.data - mu
        result = np.exp(-0.5 * np.sum(np.dot(x_minus_mu, sigma_inv) * x_minus_mu, axis=1))

        return norm_const * result


    def create_clusters(self, iter = 30):
        
        # N: Number of data points, d: Number of dimensions
        
        N, d = self.data.shape

        # To prevent singular matrix 

        epsilon = 1e-5 

        # To determine stopping criteria 

        threshold = 1e-4
        
        # Generate centroids according to K++ method if not provided

        if self.centroids.size == 0: 
            self.centroids = self.kplusplus_initialisation() 

        # Initialise covariance matrices with variance of dataset
            
        variances = np.var(self.data, axis=0)
        covar_m = [np.diag(variances) for _ in range(self.K)]
        covar_m = [cov + epsilon * np.eye(self.data.shape[1]) for cov in covar_m]

        # Equal mixing coefficients to begin with 

        pies = np.full(self.K, 1/self.K)  

        # For convergence checking

        prev_centroids = np.copy(self.centroids)
        prev_vars = np.copy(covar_m)

        for i in range(iter):

            # Responsibility calculation

            Ns = [self.multivariate_norm(self.centroids[k], covar_m[k]) for k in range(self.K)]
            total_rs = sum([Ns[k] * pies[k] for k in range(self.K)])
            Rs = [Ns[k] * pies[k] / total_rs for k in range(self.K)]

            # Setup the convergence flag            
           
            converged = False

            # Update parameters for each cluster

            for k in range(self.K):

                # Mixing coefficient 

                pie_k = np.sum(Rs[k]) / N
                
                # Means 

                centroid_k = np.sum(Rs[k][:, np.newaxis] * self.data, axis=0) / np.sum(Rs[k])
                          
                # Centre the data on k centroid 

                centred_data = self.data - centroid_k
                Rs_k = Rs[k]

                # Element-wise multiply the responsibilities with data points

                weighted_centred_data = centred_data * Rs_k[:, None]

                # Compute the covariance matrix for cluster

                covar_m_k = np.dot(weighted_centred_data.T, weighted_centred_data)

                # Normalize covariance by sum of the responsibilities

                covar_m_k /= Rs_k.sum()
                covar_m_k += (epsilon * np.eye(d))

                pies[k] = pie_k
                self.centroids[k] = centroid_k
                covar_m[k] = covar_m_k 

                if abs(prev_centroids[k] - self.centroids[k]).any() < threshold or abs(prev_vars[k] - covar_m[k]).any() < threshold:
                    converged = True

            prev_centroids = np.copy(self.centroids)
            prev_vars = np.copy(covar_m)

            if converged:
                print(f'Converged on iteration = {i}')
                break

        # Create the outputs
            
        centroids = self.centroids
        responsibility_matrix = np.column_stack(Rs)
        cluster_assignment = np.argmax(responsibility_matrix, axis=1)

        return centroids, cluster_assignment
    


    def optimise_clusters(self, num_runs = 30):
        """ 
        This function runs the entire GMM algorthm
        If centroids are provided (from k-means) then it runs once, if they arent provided it runs 30 different times
        For the latter, each run corresponds to a different initial condition for the centroids
        This process aims to find the optimal initial starting centroids to ensure convergence to the best clustering output
        It skips times when clusters are empty
        """

        if self.centroids.size == 0:

            # If we dont have centroids, we run 30 initialisations to increase likelihood of optimality 

            best_score = -999
            best_centroids = None
            best_assignment = None
        
            for _ in range(num_runs):
               
                centroids, cluster_assignment = self.create_clusters()
                
                # Calculate the score for the current run

                score = silhouettescore(self.data, cluster_assignment).score()
                
                if score > best_score:
                    best_score = score
                    best_centroids = centroids
                    best_assignment = cluster_assignment
                
        else:
            #  If we already have centroids, we only run once 
             best_centroids, best_assignment = self.create_clusters()
        
        return best_centroids, best_assignment
            
