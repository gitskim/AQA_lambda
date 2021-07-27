FROM public.ecr.aws/lambda/python:3.7

# COPY __init__.py app.py requirements.txt opts.py ./
COPY . /hello_world
WORKDIR /hello_world

RUN python3.7 -m pip install -r requirements.txt -t .
ENV PYTHONPATH "${PYTHONPATH}:/hello_world/"


# Command can be overwritten by providing a different command in the template directly.
CMD ["app.lambda_handler"]
