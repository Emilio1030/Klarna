# $DEL_BEGIN

# ####### 👇 SIMPLE SOLUTION (x86 and M1) 👇 ########
FROM python:3.8.12-buster

# tensorflow base-images are optimized: lighter than python-buster + pip install tensorflow
# FROM tensorflow/tensorflow:2.10.0

WORKDIR /prod

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY klarna klarna
COPY setup.py setup.py
COPY scripts/klarna-run /prod/klarna/scripts/klarna-run

# Add the Google Cloud SDK package repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Install the Google Cloud SDK
RUN apt-get update && apt-get install -y google-cloud-sdk
# Create the gcp_credential directory
RUN mkdir /prod/gcp_credential

# # Install NVIDIA CUDA Toolkit
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     nvidia-cuda-toolkit

# # Set environment variables (if needed)
# ENV CUDA_HOME=/usr/local/cuda
# ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64

# Copy the service account credentials JSON file to the container
COPY ../google_cloud_key/klarna-385320-6131ef0ceff2.json /prod/gcp_credential/credentials.json

RUN pip install .

# COPY Makefile Makefile
# RUN make reset_local_files

ENV GOOGLE_APPLICATION_CREDENTIALS=/prod/gcp_credential/credentials.json

# creating folders
RUN rm -rf ~/.lewagon/mlops \
    && mkdir -p ~/.lewagon/mlops/data/raw \
    && mkdir -p ~/.lewagon/mlops/data/processed \
    && mkdir -p ~/.lewagon/mlops/training_outputs/metrics \
    && mkdir -p ~/.lewagon/mlops/training_outputs/models \
    && mkdir -p ~/.lewagon/mlops/training_outputs/params


# CMD uvicorn klarna.api.fast:app --host 0.0.0.0 --port $PORT
CMD gcloud auth activate-service-account --key-file=/prod/gcp_credential/credentials.json && \
    uvicorn klarna.api.fast:app --host 0.0.0.0 --port $PORT


####### 👇 OPTIMIZED SOLUTION (x86)👇 #######

# tensorflow base-images are optimized: lighter than python-buster + pip install tensorflow
# FROM tensorflow/tensorflow:2.10.0
# # OR for apple silicon, use this base image instead
# # FROM armswdev/tensorflow-arm-neoverse:r22.09-tf-2.10.0-eigen

# WORKDIR /prod

# # We strip the requirements from useless packages like `ipykernel`, `matplotlib` etc...
# # COPY requirements_prod.txt requirements.txt
# COPY requirements.txt requirements.txt
# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt

# # COPY taxifare taxifare
# COPY klarna klarna
# COPY setup.py setup.py
# # COPY scripts/klarna-run /prod/klarna/scripts/klarna-run
# # COPY scripts/klarna-run klarna/scripts/klarna-run

# RUN pip install .

# # these refer to the new data from taxifare - I don't need it
# # COPY Makefile Makefile
# # RUN make reset_local_files

# CMD uvicorn klarna.api.fast:app --host 0.0.0.0 --port $PORT
# $DEL_END
