import numpy as np
import torch
import numpy.linalg as LA


class nC:
    def __init__(self,torch_device):
        self.torch_device = torch_device

    def neighborhoodCache(self,max_size,radius):
        """
        neighborhoodCache:
        Maintains a cache of objects (up to max_size) that all have an
        associated n-dimensional point.  The function can return the
        objects whose points are within a fixed distance (radius) of a
        query point object. (which is then also added to the cache).
        """
        self.max_size = max_size
        self.radius = radius
        #  by default, inf distances indicate empty slots for samples
        self.distances       = torch.ones((1,self.max_size),device=self.torch_device, dtype=torch.double)*float('inf')
        self.samples         = None
        # self.data            = self.max_size * [None]
        self.data            = np.empty((1,self.max_size),dtype=object)
    
        self.n               = 0
        self.last_added_ind  = 0

        get_neighborhood_fn = lambda x,x_data: self.getCachedNeighborhoodAbout(x,x_data)
        return get_neighborhood_fn

    def getCachedNeighborhoodAbout(self,x,x_data):
        if self.last_added_ind == 0:
            self.n               = 1
            self.last_added_ind  = 1
            self.distances[0,0]    = 0
            self.samples         = torch.zeros((len(x),self.max_size),device=self.torch_device, dtype=torch.double) 
            self.samples[:,0]    = x[:,0]
            self.data[0]         = x_data
            computed        = 0
        else:
            #  Calculate the distance from the new sample point x to the 
            #  most recent;y added sample already in the cache 
            # sample = self.samples[:,self.last_added_ind-1]
            # diff = x - sample
            # dist_to_last_added = LA.norm(diff)
            
            sample = self.samples[:,self.last_added_ind-1]
            diff_gpu = x.reshape(-1) - sample
            dist_to_last_added = torch.norm(diff_gpu)
            self.distances[0,self.last_added_ind-1] = 0 # will be set exactly below
            
            #  Overestimate the distances from the new sample point x to all 
            #  the other sample points by applying the triangle inequality 
            #  to dist_to_most_recent and the previously computed/estimated 
            #  distances between samples{last_index} and all the remaining
            #  points. 
            #  Note: distance(last_added_index) will be dist_to_last_added.

            self.distances      = self.distances + dist_to_last_added

            #  Only the (over)estimated distances which are greater than the 
            #  allowed radius will need to be computed exactly.
            
            indx                = torch.logical_and(self.distances > self.radius , self.distances != float('inf'))
            # indx                = self.distances > self.radius and  not np.isinf(self.distances)
            computed            = torch.sum(torch.sum(indx)).item()
            self.distances[indx]     = torch.sqrt(torch.sum(torch.square(self.samples[:,indx[0]])))
           
            if self.n < self.max_size:
                #  add x and x_data to next free slot
                self.n               = self.n + 1
                self.last_added_ind  = self.n
                self.distances[0,self.n-1]  = 0
                self.samples[:,self.n-1]    = x[0]
                self.data[0,self.n-1]         = x_data
            else:
                #  no free slot available - overwrite oldest sample
                oldest_ind              = (self.last_added_ind % self.max_size) + 1
                self.distances[0,oldest_ind-1]   = 0
                self.samples[:,oldest_ind-1]   = x.reshape(torch.numel(x))
                self.data[0,oldest_ind-1]        = x_data
                self.last_added_ind          = oldest_ind
            
            
        #  selection to return
        indx            = (self.distances <= self.radius).cpu().numpy()
        n_x             = self.samples[:,indx[0,:]]
        n_data          = self.data[indx[:]]

        return [n_x,n_data,computed]