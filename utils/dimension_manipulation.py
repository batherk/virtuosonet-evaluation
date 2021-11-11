import numpy as np
from copy import deepcopy
from sklearn.decomposition import PCA

# Projections

def scalar_projection(x,y):
    return np.dot(x, y) / np.linalg.norm(y)

def vector_projection(x,y):
    return y * np.dot(x, y) / np.dot(y, y)

# Visualization

def get_coordinates(latent_vectors, dimension_vectors, output_dimensions):
    """
    Creating understandable coordinates for visualizing latent vectors.
    Returns a numpy array with the scalar projection of each latent vector on to each dimension vectors.
    If the output_dimensions is larger than the amount of latent vectors, PCA will be used for the remaining dimensions.
    """
    if dimension_vectors.ndim == 1:
        dimension_vectors = np.expand_dims(dimension_vectors, axis=0)
    scalar_projections = np.array(
        [[scalar_projection(x, y) for x in latent_vectors] for y in dimension_vectors]).transpose()
    if output_dimensions <= dimension_vectors.shape[0]:
        return scalar_projections[:, :output_dimensions]
    else:
        remaining_dimensions = output_dimensions - dimension_vectors.shape[0]
        pca = PCA(n_components=remaining_dimensions)
        pca_coords = pca.fit_transform(X=latent_vectors)
        return np.append(scalar_projections, pca_coords, axis=1)


# Matrix

def get_dimensionality_reduction_matrix(vector):
    non_zero_indicies = np.nonzero(vector)[0]
    i_a, i_b = non_zero_indicies[-2:]
    last_row = -vector/vector[i_a]
    matrix = np.identity(len(vector))
    matrix[i_b,i_a] = vector[i_a]/vector[i_b]
    matrix[i_a] = last_row
    return matrix

def row_echelon(A, B=None):
    """ Return Row Echelon Form of matrix A """
    # if matrix A has no columns or rows,
    # it is already in REF, so we return itself
    r, c = A.shape
    if r == 0 or c == 0:
        return A, B

    if B is None:
        A = deepcopy(A)
        B = np.identity(r)

    # we search for non-zero element in the first column
    for i in range(len(A)):
        if A[i,0] != 0:
            break
    else:
        # if all elements in the first column is zero,
        # we perform REF on matrix from second column
        reduced, transformer = row_echelon(A[:,1:], B[:,1:])
        # and then add the first zero-column back
        return np.hstack([A[:,:1], reduced]), np.hstack([B[:,:1], transformer])

    # if non-zero element happens not in the first row,
    # we switch rows
    if i > 0:
        ith_row = A[i].copy()
        A[i] = A[0]
        A[0] = ith_row

        b_ith_row = B[i].copy()
        B[i] = B[0]
        B[0] = b_ith_row

    # we divide first row by first element in it
    A[0] = A[0] / A[0,0]
    B[0] = B[0] / B[0,0]

    # we subtract all subsequent rows with first row (it has 1 now as first element)
    # multiplied by the corresponding element in the first column
    A[1:] -= A[0] * A[1:,0:1]
    B[1:] -= B[0] * B[1:, 0:1]

    # we perform REF on matrix from second row, from second column
    reduced, transformer = row_echelon(A[1:,1:], B[1:,1:])

    # we add first row and first (zero) column, and return
    return np.vstack([A[:1], np.hstack([A[1:,:1], reduced]) ]), np.vstack([B[:1], np.hstack([B[1:,:1], transformer]) ])
