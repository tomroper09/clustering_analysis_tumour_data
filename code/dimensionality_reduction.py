import numpy as np
import matplotlib.pyplot as plt

class pca:
   
    """
    function for performing PCA on a given set of data for chosen number of components
    
    """
    
    def __init__(self,data,components):
        
        # initialisation
        
        self.components=components
        self.data=data
   
    def transformation(self):
       
        """
        
        Core PCA algorithm to find eigenvalues and eigenvectors of given data
        
        """
        
        #calculating mean and centering data
        
        mean=np.mean(self.data,axis=0)
        centre=self.data-mean
        
        # transpose data for calculation of covariance matrix
        
        centre_transposed=np.transpose(centre)
        
        #covariance matrix calculation
        
        cov_matrix=(np.dot(centre_transposed,centre))/len(self.data)
        
        #calculating eigenvalues and eigenvectors of covariance matrix
        
        eigenvalues,eigenvectors=np.linalg.eig(cov_matrix)
        
        #indexing eigenvalues and eigenvectors by descending size
        
        ordered_index=np.argsort(eigenvalues)[::-1]
        eigenvalues_ordered=eigenvalues[ordered_index]
        eigenvectors_ordered=eigenvectors[:,ordered_index]
        
       # calculation of cumulative explained variance to find optimal amount of PCs to remove
        
        explained_variance=eigenvalues_ordered/np.sum(eigenvalues_ordered)
        explained_variance_cumulative=np.cumsum(explained_variance)
        
        # plot of cumulative explained variance
        
        plt.figure(figsize=(10,6)) 
        plt.bar(range(1,self.data.shape[1]+1),explained_variance_cumulative)
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.xticks(range(1,10))
        plt.title('Cumulative Explained Variance per number of components')
       
        # labelling bar plot with % of cumulative explained variance
        
        for component,cev in zip(range(1,10),explained_variance_cumulative):
            plt.text(component, cev, '{:.2g}'.format(cev), ha='center', va='bottom')

        # transforming input data based on desired number of components
        
        pca_data=np.dot(centre,eigenvectors_ordered)
        pca_transformed=pca_data[:,:self.components]
        
        return eigenvalues_ordered,eigenvectors_ordered,pca_data,pca_transformed
    
    def visualisation(self,pca_transformed):
        
        """
        Visualising two largest PCs by plotting one against the other
        
        """
        
        #choose 2 most prominent features from transformed data
        
        chosen_columns=pca_transformed[:,:2]
        plt.figure(figsize=(10,6))
        plt.scatter(chosen_columns[:,0],chosen_columns[:,1])
        
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA using projected data")

