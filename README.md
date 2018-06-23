건건의 AI 연습을 위한 Repository
-----------------------------------

참고강의
-------
* 모두를 위한 머신러닝/딥러닝 강의 : https://hunkim.github.io/ml/
* 구글 머신러닝 단기집중과정 : https://developers.google.com/machine-learning/crash-course/ml-intro?hl=ko

This short guide will walk you through getting a basic one node cluster up
and running, and demonstrate some simple reads and writes. For a more-complete guide, please see the Apache Cassandra website's http://cassandra.apache.org/doc/latest/getting_started/[Getting Started Guide].

First, we'll unpack our archive:

  $ tar -zxvf apache-cassandra-$VERSION.tar.gz
  $ cd apache-cassandra-$VERSION

After that we start the server. Running the startup script with the -f argument will cause
Cassandra to remain in the foreground and log to standard out; it can be stopped with ctrl-C.

  $ bin/cassandra -f

****
