import pytest
from sparse_recommender import SparseMatrix

def test_set():
    matrix = SparseMatrix()
    matrix.set(1, 1, 5)
    assert matrix.get(1, 1) == 5

def test_get():
    matrix = SparseMatrix()
    matrix.set(2, 2, 10)
    assert matrix.get(2, 2) == 10
    assert matrix.get(1, 1) == 0  

def test_recommend():
    matrix = SparseMatrix()
    matrix.set(1, 1, 5)
    matrix.set(2, 2, 10)
    user_vector = {1: 2, 2: 1}
    recommendations = matrix.recommend(user_vector)
    assert recommendations[1] == 10  
    assert recommendations[2] == 10  

def test_add_movie():
    matrix1 = SparseMatrix()
    matrix1.set(1, 1, 5)
    matrix2 = SparseMatrix()
    matrix2.set(2, 2, 10)
    matrix1.add_movie(matrix2)
    assert matrix1.get(1, 1) == 5
    assert matrix1.get(2, 2) == 10

def test_to_dense():
    matrix = SparseMatrix()
    matrix.set(1, 1, 5)
    matrix.set(2, 2, 10)
    dense_matrix = matrix.to_dense()
    assert dense_matrix == [[5, 0], [0, 10]]

