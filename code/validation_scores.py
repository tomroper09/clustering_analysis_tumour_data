import numpy as np


"""
VALIDATION_SCORES.PY

Contains our two internal validation metrics (silhouette and S_Dbw)

VARIABLE DICTIONARY:

K = total number of clusters, k = kth cluster
Data = numpy array of features and data instances
cluster_assignment = 1-d array of which cluster each data point assigns to 
centroids = Middle point of the cluster

"""

class silhouettescore:
    

    """
        Calculation of silhouette score for a given clustering algortihm
        scores between 0 and 1 - - closest to 1 the more favourable the clustering is
        Requires: dataset (numpy) and list of cluster in which each data point 
        was allocated to after clustering algorithm
        
    """
    
    def __init__(self,data,cluster_assignment):
        
        # initialising 
        
        self.data = data
        self.cluster_assignment = cluster_assignment
    
    def score(self,noise=False):
        
        
        # initialising
        # begin silhouette score of first cluster, calculate for every cluster,
        # then find average
        
        K = np.max(self.cluster_assignment)
        silhouette_scores = []
        cluster_count = np.min(self.cluster_assignment)
        
        if noise:
            
            # in the case of noise values (in dbscan, noise cluster  = 0), begin
            # from cluster 1, avoiding noise as cluster value
            
            cluster_count = np.min(self.cluster_assignment)+1
    
        # continue until all clusters have been tested
        
        while cluster_count <= K:
            
            # begin from first cluster, continue until silhouette scores calculated
            
            # initialisation of each cluster
           
            total_cluster = self.data[self.cluster_assignment == cluster_count]
            non_cluster = self.data[self.cluster_assignment != cluster_count]

            # if cluster is empty, ignore
            
            if total_cluster.size==0:
                print('empty array. skipped')
                cluster_count+=1
                continue
            if non_cluster.size==0:
                print('empty array. skipped')
                cluster_count+=1
                continue

            # initialisation of cluster separation and cohesion
            
            total_separation=[]
            total_cohesion=[]
            
            
            for j in range(len(total_cluster)):
                
                # calculating cohesion for each point
                
                cohesion_data=np.linalg.norm(total_cluster-total_cluster[j],axis=1)
                
                # add cohesion values to array
                
                total_cohesion.append(cohesion_data)
                
                # calculating separation for each point
                
                separation_data=np.linalg.norm(non_cluster-total_cluster[j],axis=1)
                
                # add separation values to array
                
                total_separation.append(separation_data)
     

            # move on to next cluster            

            cluster_count+=1
            
            #finding average cohesion for all clusters
            
            total_cohesion=np.ravel(total_cohesion)
            cohesion=np.mean(total_cohesion)
            
            #finding average separation for all clusters
            
            total_separation=np.ravel(total_separation)
            separation=np.mean(total_separation)
            
            #calculating silhouette score
            
            silhouette_score_cluster=(separation-cohesion)/max(separation,cohesion)
            silhouette_scores.append(silhouette_score_cluster)
            
            
        # finding average silhouette score of data set
        
        silhouettescore=np.mean(silhouette_scores)
        
        print('Silhouette score  = ', silhouettescore)
            
        return silhouettescore
    



class S_Dbw():
     def __init__(self, data, cluster_assignment, centroids):
          """
          Class to calculate the S_Dbw validation score based on: 
                1. data - Data Array of features
                2. cluster_assignment - Array that denotes which cluster each data point belongs
                3. centroids - Array of the centroid location for each cluster
          It should output one single S_Dbw value
          """

          # Assign selfs and run everything on initialisation 
          self.data = data
          self.cluster_assignment = cluster_assignment
          self.centroids = centroids
          self.K = len(centroids)
          self.av_stdev = self.average_std_dev()



     def average_std_dev(self):
        """
        Function to calculate the average standard deviation of all the clusters
        """

        # Initialise Clusters and Stdev 
        total_stdev = 0

        # Calculate average stdev
        for i in range(self.K):
            std_dev_m = np.std(self.data[self.cluster_assignment == i], axis=0)
            norm = np.sqrt(np.dot(std_dev_m.T, std_dev_m))  
            total_stdev += norm
        av_stdev = np.sqrt(total_stdev) / self.K

        return av_stdev


     def single_density(self, cluster_indices):
        """
        Function to calculate the density of a cluster
        If one cluster specified, calculates the density of that cluster
        If two specified, calculates the density of the midpoint between both of them
        Calculates whether a data point is included in density by seeing if distance to midpoint is <= av_stdev
        """
        density = 0
        selected_centroids = self.centroids[cluster_indices]
        if len(cluster_indices) > 1:
            density_centroid = (selected_centroids[0] + selected_centroids[1]) /2
        else:
            density_centroid = selected_centroids
        
        for i in cluster_indices:
            cluster_data = self.data[self.cluster_assignment == i]
            for dp in cluster_data:
                if not np.linalg.norm(dp - density_centroid) > self.av_stdev:
                        density += 1

        return density


     def inter_density(self):
        """
        Function to calculate the inter-cluster density
        Its a ratio of the density of midpoint between two clusters and the max density of either of those clusters
        Its something you want to minimise, so you want the density of either cluster to be much greater than the midpoint between them
        You basically average this across all cluster pairs
        Dont include i=j where you are comparing the same cluster
        """
        tot_intra_dens = 0
        cluster_densities = []

        #  Calculate density for each cluster
        for i in range(self.K):
            cluster_densities.append(self.single_density([i]))

        #  Now calculate the intra-densities as a function of the individual cluster densities
        for i in range(self.K):
            for j in range(self.K):
                if i == j:
                        continue
                midpoint_density = self.single_density([i, j])
                max_cluster_density = max(cluster_densities[i], cluster_densities[j], 1)  
                tot_intra_dens += (midpoint_density/max_cluster_density)

        av_intra_dens = tot_intra_dens / (self.K*(self.K-1))

        return av_intra_dens


     def scatter(self):
        """
        Function to calculate the intra-cluster variance
        Its a ratio of the variance in a particular cluster divided by the variance of the total dataset
        Averaged across all clusters
        You essentially want low scatter in your clusters compared to the overall scatter
        """
        X_std_m = np.std(self.data, axis=0)
        X_var = np.sqrt(np.dot(X_std_m.T, X_std_m))
        
        tot_cluster_var = 0

        for i in range(self.K):
            cluster_data = self.data[self.cluster_assignment == i]
            cluster_std_m = np.std(cluster_data, axis=0)
            cluster_var = np.sqrt(np.dot(cluster_std_m.T, cluster_std_m))
            tot_cluster_var += cluster_var

        scatter_ratio = tot_cluster_var / X_var
        av_scatter = scatter_ratio / self.K

        return av_scatter

     def result(self):
        """
        Output function to just create the validation score
        """

        # Calculate final S_Dbw index
        computed_score = self.scatter() + self.inter_density()

        # Reference score uses individual cluster stdev rather than average, and has noise filtering 

        return computed_score
