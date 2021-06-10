FROM python

RUN pip3 install flask flask_cors gensim numpy
RUN pip3 install pyemd

ADD . /code
WORKDIR /code
CMD ["python3", "app.py"]