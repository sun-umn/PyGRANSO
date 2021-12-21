import numpy as np
import torch
import numpy.linalg as LA


class nC:
    def __init__(self,torch_device,double_precision):
        self.torch_device = torch_device
        if double_precision:
            self.torch_dtype = torch.double
        else:
            self.torch_dtype = torch.float

    def neighborhoodCache(self,max_size,radius):
        """
        neighborhoodCache:
            Maintains a cache of objects (up to max_size) that all have an
            associated n-dimensional point.  The function can return the
            objects whose points are within a fixed distance (radius) of a
            query point object. (which is then also added to the cache).

            USAGE:
                get_nbd_fn = neighborhoodCache(max_size,radius);
                [nbd_x,nbd_objects,distances_computed] = get_nbd_fn(x,object);
         
            INPUT:
                x                   an n-dimensional point (n must be the same for
                                    all calls to get_nbd_fn
         
                objects             object to cache for point x
                
            OUTPUT:
                nbd_x               n by l array of l n-dimensional points in the
                                    neighborhood of x (which also includes x)
            
                nbd_objects         cell array of length l of l objects associated 
                                    with the points in nbd_x
          
                distances_computed  number of pairwise point distances from x 
                                    computed 

            If you publish work that uses or refers to NCVX, please cite both
            NCVX and GRANSO paper:

            [1] Buyun Liang, and Ju Sun. 
                NCVX: A User-Friendly and Scalable Package for Nonconvex 
                Optimization in Machine Learning. arXiv preprint arXiv:2111.13984 (2021).
                Available at https://arxiv.org/abs/2111.13984

            [2] Frank E. Curtis, Tim Mitchell, and Michael L. Overton 
                A BFGS-SQP method for nonsmooth, nonconvex, constrained 
                optimization and its evaluation using relative minimization 
                profiles, Optimization Methods and Software, 32(1):148-181, 2017.
                Available at https://dx.doi.org/10.1080/10556788.2016.1208749
                
            Change Log:
                neighborhoodCache.m introduced in GRANSO Version 1.0
                
                Buyun Dec 20, 2021 (NCVX Version 1.0.0):
                    neighborhoodCache.py is translated from neighborhoodCache.m in GRANSO Version 1.6.4. 

            For comments/bug reports, please visit the NCVX webpage:
            https://github.com/sun-umn/NCVX
            
            NCVX Version 1.0.0, 2021, see AGPL license info below.

            =========================================================================
            |  neighborhoodCache.m                                                  |
            |  Copyright (C) 2016 Tim Mitchell                                      |
            |                                                                       |
            |  This file is originally from URTM.                                   |
            |                                                                       |
            |  URTM is free software: you can redistribute it and/or modify         |
            |  it under the terms of the GNU Affero General Public License as       |
            |  published by the Free Software Foundation, either version 3 of       |
            |  the License, or (at your option) any later version.                  |
            |                                                                       |
            |  URTM is distributed in the hope that it will be useful,              |
            |  but WITHOUT ANY WARRANTY; without even the implied warranty of       |
            |  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        |
            |  GNU Affero General Public License for more details.                  |
            |                                                                       |
            |  You should have received a copy of the GNU Affero General Public     |
            |  License along with this program.  If not, see                        |
            |  <http://www.gnu.org/licenses/agpl.html>.                             |
            =========================================================================

            =========================================================================
            |  GRANSO: GRadient-based Algorithm for Non-Smooth Optimization         |
            |  Copyright (C) 2016 Tim Mitchell                                      |
            |                                                                       |
            |  This file is translated from GRANSO.                                 |
            |                                                                       |
            |  GRANSO is free software: you can redistribute it and/or modify       |
            |  it under the terms of the GNU Affero General Public License as       |
            |  published by the Free Software Foundation, either version 3 of       |
            |  the License, or (at your option) any later version.                  |
            |                                                                       |
            |  GRANSO is distributed in the hope that it will be useful,            |
            |  but WITHOUT ANY WARRANTY; without even the implied warranty of       |
            |  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        |
            |  GNU Affero General Public License for more details.                  |
            |                                                                       |
            |  You should have received a copy of the GNU Affero General Public     |
            |  License along with this program.  If not, see                        |
            |  <http://www.gnu.org/licenses/agpl.html>.                             |
            =========================================================================

            =========================================================================
            |  NCVX (NonConVeX): A User-Friendly and Scalable Package for           |
            |  Nonconvex Optimization in Machine Learning.                          |
            |                                                                       |
            |  Copyright (C) 2021 Buyun Liang                                       |
            |                                                                       |
            |  This file is part of NCVX.                                           |
            |                                                                       |
            |  NCVX is free software: you can redistribute it and/or modify         |
            |  it under the terms of the GNU Affero General Public License as       |
            |  published by the Free Software Foundation, either version 3 of       |
            |  the License, or (at your option) any later version.                  |
            |                                                                       |
            |  GRANSO is distributed in the hope that it will be useful,            |
            |  but WITHOUT ANY WARRANTY; without even the implied warranty of       |
            |  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        |
            |  GNU Affero General Public License for more details.                  |
            |                                                                       |
            |  You should have received a copy of the GNU Affero General Public     |
            |  License along with this program.  If not, see                        |
            |  <http://www.gnu.org/licenses/agpl.html>.                             |
            =========================================================================

        """
        self.max_size = max_size
        self.radius = radius
        #  by default, inf distances indicate empty slots for samples
        self.distances       = torch.ones((1,self.max_size),device=self.torch_device, dtype=self.torch_dtype)*float('inf')
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

            self.samples         = torch.zeros((len(x),self.max_size),device=self.torch_device, dtype=self.torch_dtype) 

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