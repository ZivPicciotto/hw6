import numpy as np

def get_random_centroids(X, k):
    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    centroids = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    indexes = np.random.choice([i for i in range(len(X))], k, replace=False)
    centroids = [X[i] for i in indexes]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # make sure you return a numpy array
    return np.asarray(centroids).astype(float) 

def lp_distance(X, centroids, p=2):
    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    distances = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    image_reshaped = X[:, np.newaxis, :]
    distances = np.linalg.norm(image_reshaped-centroids, ord=p,axis=2).T
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return distances

def kmeans(X, k, p ,max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    """ See kmeans_with_itteration_count for implementation details
        added the function in order to do comparrisons later
    """
    # we simply ignore the itterations in case this function is called.
    centroids, classes, _ = kmeans_with_itteration_count(X,k,p,max_iter)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes

def kmeans_pp(X, k, p ,max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    """ See kmeans_pp_with_itteration_count for implementation details
        added the function in order to do comparrisons later
    """
    # we simply ignore the itterations in case this function is called.
    centroids, classes, _ = kmeans_pp_with_itteration_count(X,k,p,max_iter)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes

def kmeans_pp_centroid(X, k, p) -> np.ndarray:
    chosen_indexes = [np.random.choice(range(len(X)))]
    centroids = [X[chosen_indexes[0], :]]

    for _ in range(k - 1):
        mask = np.ones(len(X), dtype=bool)
        mask[chosen_indexes] = False
        remaining = X[mask, :]
        current_distances = lp_distance(remaining, centroids, p)
        min_distances = np.min(current_distances, axis=0)
        squares = min_distances**2
        probabilities = squares / np.sum(squares)

        new_centroid_index = np.random.choice(np.arange(len(remaining)), p=probabilities)
        chosen_indexes.append(np.where(mask)[0][new_centroid_index])
        centroids.append(X[chosen_indexes[-1], :])

    return np.array(centroids)


"""
this function does exactly as out original kmeans_pp implementation
however we wanted to display the number of itterations in a later section
so in order to not change the original return value of kmeans_pp we made this
helper function
"""
def kmeans_with_itteration_count(X, k, p ,max_iter=100):
    centroids = get_random_centroids(X, k)
    itterations = 0
    for _ in range(max_iter):
        itterations+=1
        distances = lp_distance(X,centroids, p)
        classes = np.argmin(distances,axis=0)
        prev = centroids
        centroids = np.array([np.mean(X[classes==i,:],axis=0) for i in range(k)])

        if np.all(prev== centroids):
            break 

    return centroids, classes, itterations

"""
this function does exactly as out original kmeans implementation
however we wanted to display the number of itterations in a later section
so in order to not change the original return value of kmeans we made this
helper function
"""
def kmeans_pp_with_itteration_count(X, k, p ,max_iter=100):
    itterations = 0
    centroids = kmeans_pp_centroid(X,k, p)
    for _ in range(max_iter):
        itterations += 1
        distances = lp_distance(X,centroids, p)
        classes = np.argmin(distances,axis=0)
        prev = centroids
        centroids = np.array([np.mean(X[classes==i,:],axis=0) for i in range(k)])

        if np.all(prev== centroids):
            break 
    return centroids, classes, itterations

