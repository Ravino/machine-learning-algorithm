FROM python:3.7


RUN apt update && apt upgrade -y
RUN apt install git nano -y
RUN pip3 install google-colab torch matplotlib seaborn tqdm sklearn mlxtend torchvision


RUN mkdir /www
RUN mkdir /www/ml


WORKDIR /www/ml


CMD [""]
