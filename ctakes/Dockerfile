FROM ubuntu:latest

# This is in accordance to : https://www.digitalocean.com/community/tutorials/how-to-install-java-with-apt-get-on-ubuntu-16-04
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/oracle-jdk8-installer;

# Fix certificate issues, found as of 
# https://bugs.launchpad.net/ubuntu/+source/ca-certificates-java/+bug/983302
RUN apt-get update && \
    apt-get install -y ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/oracle-jdk8-installer;

RUN apt-get update && apt-get install unzip

# Setup JAVA_HOME, this is useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

RUN mkdir -p /root/ctakes/
COPY ./apache-ctakes-*.tar.gz /root/ctakes/
COPY ./ctakes-resources-*.zip /root/ctakes/
COPY ./api_key.txt /root/ctakes/
COPY ./ctakes.sh /root/ctakes/

RUN cd /root/ctakes && tar -xvf apache-ctakes*.tar.gz && rm apache-ctakes*.tar.gz
RUN cd /root/ctakes && unzip -n ctakes-resources*.zip -d apache-ctakes* && rm ctakes-resources*.zip

WORKDIR /root/ctakes

ENTRYPOINT [ "./ctakes.sh" ]
