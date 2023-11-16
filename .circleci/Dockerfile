# Use the official Python image from Docker Hub as the base image
FROM python:3.8-slim
# Set working directory
WORKDIR .

# Install pytest
RUN pip install --upgrade pip && pip install pytest

# Copy
COPY .circleci/sparse_recommender.py .
COPY .circleci/test_sparse_recommender.py . 
# run

CMD [ "pytest", "test_sparse_recommender.py" ]
