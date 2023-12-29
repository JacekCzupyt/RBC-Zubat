FROM nvcr.io/nvidia/pytorch:23.02-py3

RUN mkdir /app
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

RUN git config --global --add safe.directory /app

COPY demo /app/demo

WORKDIR /app
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH "${PYTHONPATH}:."
CMD ["/bin/bash"]
