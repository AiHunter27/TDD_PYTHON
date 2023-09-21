class SparseMatrix:
    def __init__(self):
        self.matrix = {}

    def set(self, row, col, value):
        if row not in self.matrix:
            self.matrix[row] = {}
        self.matrix[row][col] = value

    def get(self, row, col):
        if row in self.matrix and col in self.matrix[row]:
            return self.matrix[row][col]
        return 0

    def recommend(self, vector):
        recommendations = {}
        for row, col_value in self.matrix.items():
            recommendation = sum(col_value.get(col, 0) * vector.get(col, 0) for col in vector)
            recommendations[row] = recommendation
        return recommendations
        
    def add_movie(self, new_matrix):
        for row, col_value in new_matrix.matrix.items():
            if row not in self.matrix:
                self.matrix[row] = {}
            for col, value in col_value.items():
                self.matrix[row][col] = value


    def to_dense(self):
        max_row = max(self.matrix.keys()) if self.matrix else 0
        max_col = max(col for row in self.matrix.values() for col in row.keys()) if self.matrix else 0
        dense_matrix = [[self.get(row, col) for col in range(1, max_col + 1)] for row in range(1, max_row + 1)]
        return dense_matrix
