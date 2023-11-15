# Use the official Python image from Docker Hub as the base image
FROM python:3.8-slim
# Set working directory
WORKDIR .
# Copy
COPY .circleci/sparse_recommender .
COPY .circleci/test_sparse_recommender . 
# run

CMD [ "pytest", "test_sparse_recommender.py" ]
