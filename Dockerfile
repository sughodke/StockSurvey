FROM python:3

RUN git clone https://b28f1e6f922df01704ef98d71d80effc1e62d19b@github.com/sughodke/StockSurvey.git
WORKDIR StockSurvey
RUN git pull

RUN pip install -r requirements.txt
EXPOSE 8080

CMD ["python", "./webservice.py"]