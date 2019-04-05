import numpy as np

class KNearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        
        for i in range(num_test): #num_test==> 500
            for j in range(num_train): #num_train==> 5000
                dists[i][j]=np.linalg.norm(X[i]-self.X_train[j])
                                            
                '''tem_sum=0
                tem_sum=(X[i][j]-self.X_train[i][j])**2
                dists[i][j]=(tem_sum)**(1/2) '''
                
        return dists
    
    def compute_distances_one_loop(self, X):
        
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test): 
            dists[i,:] = np.sqrt(np.sum((X[i,:]-self.X_train)**2, axis=1))

        return dists
    
    def compute_distances_no_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train)) 
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy.                #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        xts = np.sum(self.X_train**2, axis =1)
        xs = np.sum(X**2,axis = 1)
        dot = np.dot(X, np.transpose(self.X_train))
        dists = np.sqrt(xts[np.newaxis,:] + xs[:,np.newaxis] -2*dot)
        #########################################################################
        #                         END OF YOUR CODE                              #
        #########################################################################
        return dists
    
    def compute_L1_distances_no_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train)) 
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy.                #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        xts = np.sum(np.abs(self.X_train), axis =1)
        xs = np.sum(np.abs(X),axis = 1)
        dot = np.dot(X, np.transpose(self.X_train))
        dists = xts[np.newaxis,:] + xs[:,np.newaxis]
        #########################################################################
        #                         END OF YOUR CODE                              #
        #########################################################################
        return dists
    
    def compute_L3_distances_no_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train)) 
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy.                #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        xts = np.sum(self.X_train**3, axis =1)
        xtssqr=np.sum(self.X_train**2, axis =1)
        
        xs = np.sum(X**3,axis = 1)
        xssqr = np.sum(X**2,axis = 1)
        
        dot1 = np.dot(X, np.transpose(xtssqr))
        dot2 = np.dot(xssqr, np.transpose(self.X_train))
        dists = np.cbrt(xts[np.newaxis,:] - xs[:,np.newaxis] +3*(dot1)-3*(dot2))
        #########################################################################
        #                         END OF YOUR CODE                              #
        #########################################################################
        return dists
    
    def compute_L5_distances_no_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train)) 
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy.                #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        xts = np.sum(self.X_train**5, axis =1)
        xs = np.sum(X**5,axis = 1)
        dot = np.dot(X, np.transpose(self.X_train))
        dists = (xts[np.newaxis,:] + xs[:,np.newaxis] -2*dot)**(1/5)
        #########################################################################
        #                         END OF YOUR CODE                              #
        #########################################################################
        return dists
    
    def compute_L10_distances_no_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train)) 
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy.                #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        xts = np.sum(self.X_train**10, axis =1)
        xs = np.sum(X**10,axis = 1)
        dot = np.dot(X, np.transpose(self.X_train))
        dists = (xts[np.newaxis,:] + xs[:,np.newaxis] -2*dot)**(1/10)
        #########################################################################
        #                         END OF YOUR CODE                              #
        #########################################################################
        return dists
    
    def compute_L100_distances_no_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train)) 
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy.                #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        xts = np.sum(self.X_train**100, axis =1)
        xs = np.sum(X**100,axis = 1)
        dot = np.dot(X, np.transpose(self.X_train))
        dists = (xts[np.newaxis,:] + xs[:,np.newaxis] -2*dot)**(1/100)
        #########################################################################
        #                         END OF YOUR CODE                              #
        #########################################################################
        return dists
    
    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
             #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            closest_y = []
            
            k_near = np.argsort(dists[i])[0:k]
            #print(k_near)
            for n in k_near:
                closest_y.append(self.y_train[n])
            #print(closest_y)
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            '''min_index = np.argmin(dists) 
            y_pred[i] = self.y_train[min_index]'''
            import operator
            
            label = [0]*11
            for n in closest_y:
                label[n] +=1
            index, value = max(enumerate(label), key=operator.itemgetter(1))
            #print(index, value)
            y_pred[i] = index
            #########################################################################
            #                           END OF YOUR CODE                        #########################################################################

        return y_pred

