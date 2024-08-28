---
layout: default
title: 4. Milvus Architecture
nav_order: 4
description: ""
has_children: false
parent:  Milvus (U)
---

# Milvus Architecture
### Welcome Back

The architecture displayed here is a simplified version based on the original architecture, and it is available in the official documentation. Milvus has a decoupled storage and compute architecture. The Milvus architecture has five major components: the access layer, coordinator service, log broker, worker nodes, and decoupled storage.

### Access Layer

I will pause briefly for you to go through the components before I explain them. All right. The access layer is the one facing the end users. It handles requests and returns responses to those requests. It consists of a group of stateless proxies and provides a single endpoint address using a load balancer.

### Coordinator Service

The second major component is the coordinator service. It is the decision-making service, and its main function is to assign tasks to the worker nodes. It has four different components named route coordinator, query coordinator, data coordinator, and index coordinator. The route coordinator handles requests for creating and maintaining collections, partitions, and indexes. The query coordinator manages the topology and performs load balancing for the query nodes. The data coordinator manages the topology of the data nodes, maintains the metadata, and handles background data operations. Finally, we have the index coordinator, which manages the topology of the index nodes and oversees building and maintaining indexes.

### Worker Nodes

Let us move on to the third major component, which is the worker node. These are stateless and execute instructions received from the coordinator service. Within the worker node, we have the query node, data node, and index node. The query node fetches data from the log broker and converts it into data segments. It also loads historical data from storage and runs search operations. The data node fetches log data from the log broker, processes insert and delete operations, and stores log data in the storage component. The index node, as the name suggests, is used to build indexes.

### Storage

Let us move on to the next component, which is storage. This is responsible for data persistence. Along with the inserted vector data and indexes created for those data, Milvus stores metadata such as collection schema, status of the nodes, and checkpoints for the log broker messages, etc. Milvus uses Etcd for storing metadata and Minio to store other objects.

### Log Broker

Finally, we have the log broker, which is a publish-subscribe system. The Milvus cluster uses Apache Pulsar as the log broker, which can be easily configured to be replaced by Apache Kafka and a few other similar systems. When we insert data into the Milvus cluster, it is forwarded to the log broker from the access layer. The data present in the log broker is consumed by the worker node, which then processes and stores the data in storage. Thanks to the log broker, it is possible to replay data, maintain data integrity during crashes, reliably execute queries, and return results.

### External Dependencies

The external dependencies of Milvus include Etcd, which is a strongly consistent and distributed key-value store; Minio, a high-performance object storage system; and Apache Pulsar, a message streaming platform.

### Conclusion

So that's all for the high-level overview of the Milvus architecture. Let us look at the Milvus storage model in the next video. Thanks, and see you again.