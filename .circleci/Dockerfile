# Use the official Python image from Docker Hub as the base image
FROM python:3.8-slim


RUN apt-get update && apt-get install -y python3-pip && pip3 install pytest



# Set working directory
WORKDIR .



# Copy
COPY .circleci/sparse_recommender.py .
COPY .circleci/test_sparse_recommender.py . 
# run

CMD [ "pytest", "test_sparse_recommender.py" ]
