rdua@rdua-ltmh889 github % docker run \
    --restart always \
    --publish=7474:7474 --publish=7687:7687 \
    neo4j:5.26.0
Unable to find image 'neo4j:5.26.0' locally
5.26.0: Pulling from library/neo4j
879a6187682f: Pull complete 
b8bfdadce121: Pull complete 
46120a0c7324: Pull complete 
f549d6f5d7d3: Pull complete 
c783cffde723: Pull complete 
4f4fb700ef54: Pull complete 
Digest: sha256:411e532d5c9892e7c955b4e631da423af41ede6a683a0de02c93876b7509c2db
Status: Downloaded newer image for neo4j:5.26.0
2025-01-07 14:37:21.506+0000 INFO  Logging config in use: File '/var/lib/neo4j/conf/user-logs.xml'
2025-01-07 14:37:21.514+0000 INFO  Starting...
2025-01-07 14:37:22.057+0000 INFO  This instance is ServerId{a41b6de7} (a41b6de7-886d-4633-9ab2-fe12e1e00035)
2025-01-07 14:37:22.681+0000 INFO  ======== Neo4j 5.26.0 ========
2025-01-07 14:37:24.026+0000 INFO  Anonymous Usage Data is being sent to Neo4j, see https://neo4j.com/docs/usage-data/
2025-01-07 14:37:24.177+0000 INFO  Bolt enabled on 0.0.0.0:7687.
2025-01-07 14:37:24.553+0000 INFO  HTTP enabled on 0.0.0.0:7474.
2025-01-07 14:37:24.553+0000 INFO  Remote interface available at http://localhost:7474/
2025-01-07 14:37:24.555+0000 INFO  id: 8528491FB41FA7E41EE891CAF4DA633A916FD61395EFFE7A0ED56A61F3218E61
2025-01-07 14:37:24.555+0000 INFO  name: system
2025-01-07 14:37:24.555+0000 INFO  creationDate: 2025-01-07T14:37:23.363Z
2025-01-07 14:37:24.555+0000 INFO  Started.

