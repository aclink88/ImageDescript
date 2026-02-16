# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY . .

# Install any needed packages specified in pyproject.toml
# We will need to use a build backend that can read pyproject.toml, like pip.
RUN pip install --no-cache-dir .

# Define environment variable
ENV NAME World

# Run the application (example, will be changed later)
CMD ["python", "src/parser/excel_parser.py"]
