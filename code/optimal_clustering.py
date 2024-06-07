import numpy as np
import matplotlib.pyplot as plt
from clustering_techniques import knn
from clustering_techniques import gmm
from validation_scores import silhouettescore
from validation_scores import S_Dbw

class INERTIA:
    
    """
    
    Calculates inertia of a data set and its clusters
    Requires a numpy data set, with a 1-d array of cluster assignments
    """
    
    def __init__(self,data,clusters):
        
        # initialisation
        
        self
        self.data=data
        self.clusters=clusters
        
    def result(self):
        
        # initialisation
        
        inertia=[]
        current_cluster=np.min(self.clusters)
        final_cluster=np.max(self.clusters)
        
        # calculate inertia for each point, going by cluster 
        
        while current_cluster<=final_cluster:
            
            # points within same cluster put in array
            
            total_cluster=self.data[self.clusters==current_cluster]
            
            # initialise centroid point of cluster 
            
            centroid=np.mean(total_cluster,axis=0)
            
            for row in total_cluster:
                
                # inertia value for each point calculated
                
                distance=np.sum((np.linalg.norm(row-centroid))**2)
                
                # append into array
                
                inertia.append(distance)
            
            # move to next cluster
            
            current_cluster+=1
            
        
        # result is sum of all inertias 
        
        return np.sum(inertia)




class scoring:
    
    """

    Calculates means squared measure between clusters, for SPECIFIC clustering techniques, created from
    a range 'k_range' of cluster values, then plots the outputted inertia for each
    cluster value
    
    Additionally calculate silhouette and S_dbw score for each cluster

    """
    
    
    
    def __init__(self,data,k_range=range(1,31)):
        
        """
        
        data = inputted data
        k_range = range of cluster numbers to check
        """
        
        # initialisaion
        
        
        self.data=data
        self.k_range=k_range
        

    def score_kmeans(self):
        
        """
        knn attribute means squared calculation
        Silhouette score and S_Dbw scores included
        """
        
        # initialisation
        
        inertia_values=[]
        silhouettes=[]
        sdbws=[]
        inertia_difference=np.zeros(len(self.k_range))
        
        for k in self.k_range:
            
            # initiate KNN algorithm for each k value in range
            
            KNN=knn(self.data,k)
            best_centroids,best_assignment=KNN.optimise_clusters()
            
            # calculating inertia of points given cluster arrangement caluclated prior
            inertia=INERTIA(self.data,best_assignment).result()
            
            # add all inertia values to same array
            
            inertia_values.append(inertia)
            
            # calculation of silhouette score
            
            s_score=silhouettescore(self.data,best_assignment).score()
           
            # add all silhouettes to same array
            
            silhouettes.append(s_score)
            
            # calculation of S_Dbw score
            
            sdbw=S_Dbw(self.data,best_assignment,best_centroids).result()
            
            # add all S_Dbw scores to same array
            
            sdbws.append(sdbw)
        
        #inertia difference to use when determining optimal k
        
        inertia_difference[0]=inertia_values[0]
        
        for i in range(1,len(inertia_values)):
            
            # subtract current inertia from previous inertia
            
            inertia_difference[i]=inertia_values[i-1]-inertia_values[i]
            
            
            
        #plotting means squared graph
        
        plt.figure(figsize=(10,6))
        plt.plot(self.k_range,inertia_values)
        plt.xlabel('Number of Clusters, k')
        plt.ylabel('Total Inertia')
        
        # labelling each k value
        
        for k,inertia in zip(self.k_range,inertia_values):
            plt.text(k,inertia, '{:.2g}'.format(k), ha='center', va='bottom')
        
        #plotting silhouette score
        
        plt.figure(figsize=(10,6))
        plt.plot(self.k_range,silhouettes)

        plt.yticks([0,0.2,0.4,0.6,0.8,1])
        plt.xlabel('Number of Clusters, k')
        plt.ylabel('Silhouette score')
        plt.title('Change in silhouette score as number of clusters increases (k-means)')
        
        plt.figure(figsize=(10,6))
        plt.plot(self.k_range,sdbws)

        plt.yticks([0,0.2,0.4,0.6,0.8,1])
        plt.xlabel('Number of Clusters, k')
        plt.ylabel('S_Dbw score')
        plt.title('Change in S_Dbw score as number of clusters increases (k-means)')
        
        return inertia_values,inertia_difference,silhouettes,sdbws


   
    def score_gmm(self):
        
        """
        gmm scores squared distances, S_Dbw and Sil
        """
        
        # initialisation
        
        inertia_values=[]
        silhouettes=[]
        sdbws=[]
        inertia_difference=np.zeros(len(self.k_range))
        actuals=[]
        
        for k in self.k_range:
            
            #calculation of cluster assignment using knn for k clusters
            
            best_centroids_knn,best_assignment_knn=knn(self.data,k).optimise_clusters()
            
            # optimise knn cluster assignment using gmm
            
            best_centroids,best_assignment=gmm(self.data,k,best_centroids_knn).optimise_clusters()
            
            # calculating inertia of points given cluster arrangement calculated prior
            inertia=INERTIA(self.data,best_assignment).result()
            
            # add all inertia values together
            
            inertia_values.append(inertia)
            
            # calculation of silhouette score
            
            s_score=silhouettescore(self.data,best_assignment).score()
            
            # adding all silhouette scores to same array
            
            silhouettes.append(s_score)

            # calculation of S_Dbw score
            sdbw=S_Dbw(self.data,best_assignment,best_centroids).result()
            
            # adding all S_Dbw scores to same array
            
            sdbws.append(sdbw)
            
            # calculating inertia differences
 
            inertia_difference[0]=inertia_values[0]
            
            for i in range(1,len(inertia_values)):
                
                # substraction of current inertia from previous
                
                inertia_difference[i]=inertia_values[i-1]-inertia_values[i]
                
                
            
        #plotting means squared graph
        
        plt.figure(figsize=(10,6))
        plt.plot(self.k_range,inertia_values)
        plt.xlabel('Amount of clusters, k')
        plt.ylabel('Inertia')
        plt.title('Means squared results (GMM)')
        
        #plotting silhouette score
        
        plt.figure(figsize=(10,6))
        plt.plot(self.k_range,silhouettes)
        plt.yticks([0,0.2,0.4,0.6,0.8,1])
        plt.xlabel('Amount of clusters, k')
        plt.ylabel('Silhouette score')
        plt.title('Change in silhouette score as number of clusters increases (GMM)')

        #plotting S_Dbw score
    
        plt.figure(figsize=(10,6))
        plt.plot(self.k_range,sdbws)
        plt.yticks([0,0.2,0.4,0.6,0.8,1])
        plt.xlabel('Amount of clusters, k')
        plt.ylabel('S_Dbw score')
        plt.title('Change in S_Dbw score as number of clusters increases (GMM)')
        
        return inertia_values,inertia_difference,silhouettes,sdbws,actuals
