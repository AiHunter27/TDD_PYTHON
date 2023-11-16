# Use the official Python image from Docker Hub as the base image
FROM python:3.8-slim

# Install pytest
RUN pip install --upgrade pip && pip install pytest



# Set working directory
WORKDIR .



# Copy
COPY .circleci/sparse_recommender.py .
COPY .circleci/test_sparse_recommender.py . 
# run

CMD [ "pytest", "test_sparse_recommender.py" ]
