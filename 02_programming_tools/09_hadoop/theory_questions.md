# Hadoop Interview Questions - Theory Questions

## Question 1

**What is Hadoop and what are its core components?**

### Answer

**Definition**: Hadoop is an open-source framework for distributed storage and processing of large datasets across clusters of computers using simple programming models.

### Core Components

| Component | Description |
|-----------|-------------|
| **HDFS** | Hadoop Distributed File System - storage layer |
| **MapReduce** | Processing engine for batch processing |
| **YARN** | Yet Another Resource Negotiator - resource management |
| **Hadoop Common** | Utilities and libraries |

### HDFS Architecture

| Component | Role |
|-----------|------|
| NameNode | Master - manages metadata |
| DataNode | Slave - stores actual data |
| Secondary NameNode | Checkpoint for NameNode |

### Python Code Example (using hdfs library)
```python
from hdfs import InsecureClient

# Connect to HDFS
client = InsecureClient('http://namenode:50070', user='hadoop')

# List directory contents
files = client.list('/')
print(f"Root files: {files}")

# Upload file
client.upload('/user/data/', 'local_file.txt')

# Download file
client.download('/user/data/file.txt', 'local_copy.txt')

# Read file content
with client.read('/user/data/file.txt') as reader:
    content = reader.read()
```

---

## Question 2

**Explain the HDFS architecture in detail.**

### Answer

### HDFS Components

| Component | Function |
|-----------|----------|
| **NameNode** | Stores metadata (file names, block locations) |
| **DataNode** | Stores actual data blocks |
| **Block** | Default 128MB chunks |
| **Replication** | Default 3 copies for fault tolerance |

### Data Flow

| Operation | Flow |
|-----------|------|
| Write | Client → NameNode → DataNodes (pipeline) |
| Read | Client → NameNode → DataNode |

### Python Code Example
```python
from hdfs import InsecureClient

client = InsecureClient('http://namenode:50070', user='hadoop')

# Check file status
status = client.status('/user/data/large_file.csv')
print(f"File size: {status['length']} bytes")
print(f"Replication: {status['replication']}")
print(f"Block size: {status['blockSize']}")

# Create directory
client.makedirs('/user/new_directory')

# Delete file
client.delete('/user/old_file.txt')

# Rename/Move
client.rename('/user/old_path', '/user/new_path')

# Set replication factor
client.set_replication('/user/important_file.txt', replication=5)
```

### HDFS Design Principles
- **Write-once, read-many**: Optimized for batch processing
- **Large files**: Designed for GB/TB size files
- **Streaming access**: Sequential reads preferred
- **Commodity hardware**: Fault tolerance through replication

---

## Question 3

**What is MapReduce and how does it work?**

### Answer

**Definition**: MapReduce is a programming model for processing large datasets in parallel across a Hadoop cluster.

### Phases

| Phase | Description |
|-------|-------------|
| **Map** | Process input, emit key-value pairs |
| **Shuffle** | Group by key, sort |
| **Reduce** | Aggregate values for each key |

### Python Code Example (using mrjob)
```python
from mrjob.job import MRJob
from mrjob.step import MRStep

# Word Count Example
class WordCount(MRJob):
    
    def mapper(self, _, line):
        """Map phase: emit (word, 1) for each word"""
        for word in line.strip().split():
            yield word.lower(), 1
    
    def reducer(self, word, counts):
        """Reduce phase: sum counts for each word"""
        yield word, sum(counts)

# Run: python wordcount.py input.txt

# Advanced: Multi-step MapReduce
class TopWords(MRJob):
    
    def steps(self):
        return [
            MRStep(mapper=self.mapper_count,
                   reducer=self.reducer_count),
            MRStep(reducer=self.reducer_top)
        ]
    
    def mapper_count(self, _, line):
        for word in line.strip().split():
            yield word.lower(), 1
    
    def reducer_count(self, word, counts):
        yield None, (sum(counts), word)
    
    def reducer_top(self, _, word_counts):
        # Get top 10 words
        top_10 = sorted(word_counts, reverse=True)[:10]
        for count, word in top_10:
            yield word, count

if __name__ == '__main__':
    WordCount.run()
```

---

## Question 4

**What is YARN and what are its components?**

### Answer

**Definition**: YARN (Yet Another Resource Negotiator) is Hadoop's resource management layer that separates resource management from processing.

### Components

| Component | Description |
|-----------|-------------|
| **ResourceManager** | Master - allocates resources |
| **NodeManager** | Slave - manages containers on each node |
| **ApplicationMaster** | Per-app coordinator |
| **Container** | Resource allocation unit |

### YARN vs MapReduce 1.0

| Aspect | MapReduce 1.0 | YARN |
|--------|---------------|------|
| Resource Management | JobTracker | ResourceManager |
| Processing | MapReduce only | Multiple frameworks |
| Scalability | ~4000 nodes | ~10000 nodes |

### Python Code Example
```python
# Using YARN REST API
import requests
import json

YARN_RM_URL = "http://resourcemanager:8088"

# Get cluster info
def get_cluster_info():
    response = requests.get(f"{YARN_RM_URL}/ws/v1/cluster/info")
    return response.json()

# Get cluster metrics
def get_cluster_metrics():
    response = requests.get(f"{YARN_RM_URL}/ws/v1/cluster/metrics")
    metrics = response.json()['clusterMetrics']
    return {
        'activeNodes': metrics['activeNodes'],
        'totalMemory': metrics['totalMB'],
        'availableMemory': metrics['availableMB'],
        'appsRunning': metrics['appsRunning']
    }

# List applications
def list_applications(state='RUNNING'):
    response = requests.get(
        f"{YARN_RM_URL}/ws/v1/cluster/apps",
        params={'state': state}
    )
    return response.json()

# Kill application
def kill_application(app_id):
    response = requests.put(
        f"{YARN_RM_URL}/ws/v1/cluster/apps/{app_id}/state",
        json={'state': 'KILLED'}
    )
    return response.status_code == 200

print(get_cluster_metrics())
```

---

## Question 5

**What is the difference between HDFS and traditional file systems?**

### Answer

### Comparison

| Aspect | HDFS | Traditional FS |
|--------|------|----------------|
| **File Size** | GB to TB | KB to GB |
| **Access Pattern** | Write-once, read-many | Random read/write |
| **Block Size** | 128MB default | 4KB typical |
| **Replication** | Built-in (3x) | RAID or manual |
| **Fault Tolerance** | Automatic | Hardware dependent |
| **Scalability** | Thousands of nodes | Single machine |

### When to Use HDFS

| Use Case | HDFS Suitable |
|----------|---------------|
| Log analysis | ✅ Yes |
| Random access | ❌ No |
| Large batch processing | ✅ Yes |
| Low-latency queries | ❌ No |
| Data archival | ✅ Yes |

### Python Code Example
```python
from hdfs import InsecureClient
import os

# HDFS operations vs local file system
class FileSystemComparison:
    def __init__(self, hdfs_url, hdfs_user):
        self.hdfs = InsecureClient(hdfs_url, user=hdfs_user)
    
    def upload_to_hdfs(self, local_path, hdfs_path):
        """Upload local file to HDFS"""
        self.hdfs.upload(hdfs_path, local_path)
        
        # Verify
        status = self.hdfs.status(hdfs_path)
        print(f"Uploaded: {status['length']} bytes")
        print(f"Block size: {status['blockSize']} bytes")
        print(f"Replication: {status['replication']}")
    
    def compare_read_performance(self, hdfs_path, local_path):
        """Compare read performance"""
        import time
        
        # HDFS read
        start = time.time()
        with self.hdfs.read(hdfs_path) as reader:
            hdfs_data = reader.read()
        hdfs_time = time.time() - start
        
        # Local read
        start = time.time()
        with open(local_path, 'rb') as f:
            local_data = f.read()
        local_time = time.time() - start
        
        print(f"HDFS read: {hdfs_time:.2f}s")
        print(f"Local read: {local_time:.2f}s")
```

---

## Question 6

**Explain data replication in HDFS.**

### Answer

### Replication Strategy

| Factor | Default | Description |
|--------|---------|-------------|
| Replication Factor | 3 | Number of copies |
| Rack Awareness | Enabled | Distributes across racks |
| Block Placement | Optimized | 1 local, 2 remote |

### Block Placement Policy
1. First replica: Same node as writer (or random if external)
2. Second replica: Different rack
3. Third replica: Same rack as second, different node

### Python Code Example
```python
from hdfs import InsecureClient

client = InsecureClient('http://namenode:50070', user='hadoop')

# Set replication for specific file
def set_replication(path, factor):
    """Change replication factor"""
    client.set_replication(path, replication=factor)
    status = client.status(path)
    print(f"New replication: {status['replication']}")

# Check replication status
def check_replication(path):
    """Check if file is under-replicated"""
    status = client.status(path)
    current = status['replication']
    
    # Using WebHDFS to get block info
    import requests
    response = requests.get(
        f'http://namenode:50070/webhdfs/v1{path}?op=GETFILEBLOCKLOCATIONS'
    )
    blocks = response.json()
    
    for block in blocks.get('BlockLocations', {}).get('BlockLocation', []):
        actual_replicas = len(block.get('hosts', []))
        if actual_replicas < current:
            print(f"Block under-replicated: {actual_replicas}/{current}")

# Different replication for different data types
def upload_with_replication(local_path, hdfs_path, data_type):
    """Upload with appropriate replication"""
    replication_config = {
        'critical': 5,
        'normal': 3,
        'temporary': 1
    }
    
    client.upload(hdfs_path, local_path)
    client.set_replication(hdfs_path, replication_config.get(data_type, 3))
```

---

## Question 7

**What is NameNode federation and High Availability?**

### Answer

### NameNode High Availability (HA)

| Component | Role |
|-----------|------|
| Active NameNode | Handles all client operations |
| Standby NameNode | Ready to take over |
| JournalNodes | Store edit logs (quorum) |
| ZooKeeper | Automatic failover |

### Federation

| Feature | Description |
|---------|-------------|
| Multiple NameNodes | Each manages namespace portion |
| Block Pools | Each NameNode has its own |
| Scalability | Horizontal namespace scaling |

### Python Code Example
```python
import requests
from kazoo.client import KazooClient

class HDFSHAClient:
    def __init__(self, namenodes, zk_hosts):
        self.namenodes = namenodes  # List of NameNode URLs
        self.zk = KazooClient(hosts=zk_hosts)
        self.zk.start()
    
    def get_active_namenode(self):
        """Find active NameNode"""
        for nn in self.namenodes:
            try:
                response = requests.get(f"{nn}/jmx?qry=Hadoop:service=NameNode,name=NameNodeStatus")
                status = response.json()['beans'][0]['State']
                if status == 'active':
                    return nn
            except:
                continue
        return None
    
    def check_health(self):
        """Check cluster health"""
        active = self.get_active_namenode()
        if not active:
            return {'status': 'CRITICAL', 'message': 'No active NameNode'}
        
        response = requests.get(f"{active}/jmx?qry=Hadoop:service=NameNode,name=FSNamesystem")
        metrics = response.json()['beans'][0]
        
        return {
            'status': 'OK',
            'activeNN': active,
            'totalBlocks': metrics['BlocksTotal'],
            'missingBlocks': metrics['MissingBlocks'],
            'underReplicatedBlocks': metrics['UnderReplicatedBlocks']
        }
    
    def trigger_failover(self):
        """Manual failover (use with caution)"""
        # This would typically use hdfs haadmin command
        import subprocess
        result = subprocess.run(
            ['hdfs', 'haadmin', '-failover', 'nn1', 'nn2'],
            capture_output=True, text=True
        )
        return result.returncode == 0
```

---

## Question 8

**What are the different input formats in Hadoop?**

### Answer

### Common Input Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| TextInputFormat | Line-by-line text | Log files |
| KeyValueTextInputFormat | Tab-separated K-V | Structured text |
| SequenceFileInputFormat | Binary K-V | Intermediate data |
| NLineInputFormat | N lines per split | Fixed-size splits |
| CombineFileInputFormat | Combine small files | Many small files |

### Python Code Example (mrjob)
```python
from mrjob.job import MRJob
from mrjob.protocol import JSONValueProtocol, RawValueProtocol
import json
import csv

# Processing different input formats

# 1. JSON Lines format
class ProcessJSON(MRJob):
    INPUT_PROTOCOL = RawValueProtocol
    
    def mapper(self, _, line):
        record = json.loads(line)
        yield record['category'], record['amount']
    
    def reducer(self, category, amounts):
        yield category, sum(amounts)

# 2. CSV format
class ProcessCSV(MRJob):
    
    def mapper(self, _, line):
        row = next(csv.reader([line]))
        if len(row) >= 3:
            yield row[0], float(row[2])  # category, amount
    
    def reducer(self, key, values):
        yield key, sum(values)

# 3. Custom delimiter
class ProcessCustomDelimiter(MRJob):
    
    def mapper(self, _, line):
        fields = line.split('|')  # Pipe delimiter
        yield fields[0], 1
    
    def reducer(self, key, counts):
        yield key, sum(counts)

# 4. Multi-line records (using combiner)
class ProcessMultiLine(MRJob):
    
    def mapper_init(self):
        self.buffer = []
    
    def mapper(self, _, line):
        if line.startswith('---'):  # Record separator
            if self.buffer:
                record = '\n'.join(self.buffer)
                yield 'record', record
                self.buffer = []
        else:
            self.buffer.append(line)
    
    def mapper_final(self):
        if self.buffer:
            yield 'record', '\n'.join(self.buffer)

if __name__ == '__main__':
    ProcessJSON.run()
```

---

## Question 9

**Explain the Hadoop ecosystem components.**

### Answer

### Ecosystem Overview

| Component | Category | Purpose |
|-----------|----------|---------|
| **Hive** | SQL | SQL-like queries |
| **Pig** | Scripting | Data flow language |
| **HBase** | NoSQL | Real-time random access |
| **Spark** | Processing | In-memory processing |
| **Sqoop** | Data Transfer | RDBMS ↔ HDFS |
| **Flume** | Ingestion | Log collection |
| **Kafka** | Streaming | Message queue |
| **Oozie** | Workflow | Job scheduling |
| **ZooKeeper** | Coordination | Distributed coordination |

### Python Code Example
```python
# Working with Hadoop ecosystem

# 1. Hive with PyHive
from pyhive import hive

conn = hive.Connection(host='hiveserver', port=10000, database='default')
cursor = conn.cursor()

cursor.execute('SELECT * FROM sales LIMIT 10')
for row in cursor.fetchall():
    print(row)

# 2. HBase with happybase
import happybase

connection = happybase.Connection('hbase-master')
table = connection.table('users')

# Put data
table.put(b'user1', {b'info:name': b'John', b'info:age': b'30'})

# Get data
row = table.row(b'user1')
print(row)

# Scan
for key, data in table.scan(row_prefix=b'user'):
    print(key, data)

# 3. Sqoop-like data transfer
import subprocess

def sqoop_import(jdbc_url, table, target_dir, username, password):
    """Import from RDBMS to HDFS"""
    cmd = [
        'sqoop', 'import',
        '--connect', jdbc_url,
        '--username', username,
        '--password', password,
        '--table', table,
        '--target-dir', target_dir,
        '--num-mappers', '4'
    ]
    subprocess.run(cmd, check=True)

# 4. Oozie workflow submission
def submit_oozie_workflow(oozie_url, properties):
    """Submit Oozie workflow"""
    import requests
    
    response = requests.post(
        f"{oozie_url}/v1/jobs",
        params={'action': 'start'},
        data=properties
    )
    return response.json()['id']
```

---

## Question 10

**What is speculative execution in Hadoop?**

### Answer

**Definition**: Speculative execution launches backup copies of slow-running tasks to reduce job completion time.

### How It Works

| Step | Description |
|------|-------------|
| Monitor | Track task progress |
| Detect | Identify slow tasks |
| Launch | Start backup task |
| Kill | Terminate slower one |

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mapreduce.map.speculative` | true | Enable for mappers |
| `mapreduce.reduce.speculative` | true | Enable for reducers |

### Python Code Example
```python
from mrjob.job import MRJob
from mrjob.step import MRStep
import time
import random

class JobWithSpeculative(MRJob):
    """Example showing when speculative execution helps"""
    
    JOBCONF = {
        'mapreduce.map.speculative': 'true',
        'mapreduce.reduce.speculative': 'true',
        'mapreduce.job.speculative.slowtaskthreshold': '1.0',  # 100% slower than average
    }
    
    def mapper(self, _, line):
        # Simulate variable processing time
        # In real scenarios, this could be due to:
        # - Bad hardware
        # - Network issues
        # - Data skew
        
        if random.random() < 0.01:  # 1% chance of being slow
            time.sleep(10)  # Slow task
        
        for word in line.split():
            yield word, 1
    
    def reducer(self, word, counts):
        yield word, sum(counts)

# Monitoring speculative execution
class SpeculativeMonitor:
    def __init__(self, yarn_url):
        self.yarn_url = yarn_url
    
    def get_speculative_tasks(self, app_id):
        """Get info about speculative task attempts"""
        import requests
        
        response = requests.get(
            f"{self.yarn_url}/ws/v1/history/mapreduce/jobs/{app_id}/tasks"
        )
        tasks = response.json()['tasks']['task']
        
        speculative = []
        for task in tasks:
            if task.get('successfulAttempt') != task.get('id') + '_0':
                # Original attempt wasn't successful
                speculative.append({
                    'taskId': task['id'],
                    'type': task['type'],
                    'elapsedTime': task['elapsedTime']
                })
        
        return speculative

# When to disable speculative execution
"""
Disable when:
1. Tasks have side effects (writing to external systems)
2. Tasks are expensive to restart
3. Cluster is heavily loaded
4. Tasks are non-idempotent
"""

if __name__ == '__main__':
    JobWithSpeculative.run()
```

### Best Practices
- Enable for CPU-bound tasks
- Disable for tasks with external side effects
- Monitor cluster resources
- Consider data locality impact


---

# --- Missing Questions Restored from Source (Q11-Q30) ---

## Question 11

**Explain howApache Flumehelps withlog and event datacollection forHadoop.**

**Answer:**

### Definition
Apache Flume is a distributed, reliable service for collecting, aggregating, and moving **large amounts of log/event data** into HDFS or other centralized data stores.

### Architecture

| Component | Role |
|-----------|------|
| **Source** | Ingests data (log files, HTTP, Kafka, syslog) |
| **Channel** | Buffers between source and sink (memory or file) |
| **Sink** | Writes to destination (HDFS, HBase, Kafka) |
| **Agent** | JVM process containing source → channel → sink |

### Data Flow
```
Web Servers ──┐
               ├──▶ Flume Agent (Source → Channel → Sink) ──▶ HDFS
App Logs ─────┘
```

### Configuration Example
```properties
# flume-conf.properties
agent.sources = weblog
agent.channels = memchannel
agent.sinks = hdfssink

# Source: tail log files
agent.sources.weblog.type = exec
agent.sources.weblog.command = tail -F /var/log/apache/access.log
agent.sources.weblog.channels = memchannel

# Channel: in-memory buffer
agent.channels.memchannel.type = memory
agent.channels.memchannel.capacity = 10000

# Sink: write to HDFS
agent.sinks.hdfssink.type = hdfs
agent.sinks.hdfssink.hdfs.path = /user/logs/%Y/%m/%d
agent.sinks.hdfssink.hdfs.fileType = DataStream
agent.sinks.hdfssink.channel = memchannel
```

### Interview Tip
Flume provides **at-least-once delivery** with file channels (durable) or **best-effort** with memory channels (faster). In modern stacks, Apache Kafka has largely replaced Flume for event streaming, but Flume is still used for direct-to-HDFS ingestion.

---

## Question 12

**What isApache Sqoopand how does it interact withHadoop?**

**Answer:**

### Definition
Apache Sqoop (SQL-to-Hadoop) is a tool for efficiently transferring **bulk data between relational databases (RDBMS) and Hadoop** (HDFS, Hive, HBase).

### Operations

| Operation | Direction | Command |
|-----------|-----------|--------|
| **Import** | RDBMS → HDFS | `sqoop import` |
| **Export** | HDFS → RDBMS | `sqoop export` |
| **List databases** | Check RDBMS | `sqoop list-databases` |
| **List tables** | Check RDBMS | `sqoop list-tables` |

### Code Examples
```bash
# Import from MySQL to HDFS
sqoop import \
  --connect jdbc:mysql://dbserver/mydb \
  --username root --password secret \
  --table employees \
  --target-dir /user/hadoop/employees \
  --num-mappers 4 \
  --split-by emp_id

# Import into Hive table
sqoop import \
  --connect jdbc:mysql://dbserver/mydb \
  --table sales \
  --hive-import \
  --hive-table sales_data \
  --incremental append \
  --check-column id \
  --last-value 1000

# Export from HDFS to MySQL
sqoop export \
  --connect jdbc:mysql://dbserver/mydb \
  --table results \
  --export-dir /user/hadoop/output \
  --input-fields-terminated-by ','
```

### How It Works
- Uses **MapReduce** for parallel data transfer
- Each mapper handles a portion of the data
- `--split-by` determines how to partition the import across mappers
- Supports incremental imports (`--incremental append/lastmodified`)

### Interview Tip
Sqoop is being deprecated in favor of **Apache Spark JDBC** connectors and tools like **Apache NiFi**. However, it's still widely used in legacy Hadoop clusters. Key advantage: it uses MapReduce for parallelism, so a 4-mapper import is ~4x faster than a single JDBC connection.

---

## Question 13

**How doesApache Ooziehelp inworkflow schedulinginHadoop?**

**Answer:**

### Definition
Apache Oozie is a **workflow scheduler** for managing and orchestrating Hadoop jobs (MapReduce, Pig, Hive, Spark, etc.) as directed acyclic graphs (DAGs).

### Workflow Types

| Type | Purpose | Trigger |
|------|---------|--------|
| **Workflow** | Sequential/parallel job execution | Manual or programmatic |
| **Coordinator** | Time/data-triggered recurring workflows | Cron-like schedule |
| **Bundle** | Group of coordinators | Manage related pipelines |

### Workflow XML Example
```xml
<workflow-app name="etl-pipeline" xmlns="uri:oozie:workflow:0.5">
    <start to="extract"/>
    
    <action name="extract">
        <sqoop xmlns="uri:oozie:sqoop-action:0.2">
            <command>import --connect jdbc:mysql://db/sales --table orders --target-dir /data/raw</command>
        </sqoop>
        <ok to="transform"/>
        <error to="fail"/>
    </action>
    
    <action name="transform">
        <hive xmlns="uri:oozie:hive-action:0.2">
            <script>transform.hql</script>
        </hive>
        <ok to="load"/>
        <error to="fail"/>
    </action>
    
    <action name="load">
        <spark xmlns="uri:oozie:spark-action:0.1">
            <master>yarn</master>
            <jar>analytics.jar</jar>
        </spark>
        <ok to="end"/>
        <error to="fail"/>
    </action>
    
    <kill name="fail"><message>Pipeline failed</message></kill>
    <end name="end"/>
</workflow-app>
```

### Coordinator (Scheduled)
```xml
<coordinator-app name="daily-etl" frequency="${coord:days(1)}">
    <action>
        <workflow>
            <app-path>/user/oozie/etl-pipeline</app-path>
        </workflow>
    </action>
</coordinator-app>
```

### Interview Tip
Oozie is Hadoop-native but verbose (XML-based). Modern alternatives include **Apache Airflow** (Python DAGs, more flexible), **Luigi**, and **Prefect**. However, Oozie integrates tightly with Hadoop security (Kerberos) and YARN, making it still relevant in enterprise Hadoop clusters.

---

## Question 14

**What isApache ZooKeeperand why is itimportantforHadoop?**

**Answer:**

### Definition
Apache ZooKeeper is a centralized service for **distributed coordination** — it provides configuration management, naming, synchronization, and group services for distributed systems.

### Role in Hadoop

| Function | Description |
|----------|-------------|
| **Leader election** | NameNode HA (active/standby failover) |
| **Configuration management** | Centralized config for cluster services |
| **Distributed locking** | Prevent concurrent modifications |
| **Service discovery** | Track which services are alive |
| **Barrier synchronization** | Coordinate distributed processes |

### Architecture
```
Client 1 ──┐                    ┌─ ZK Node 1 (Leader)
Client 2 ──├─▶ ZooKeeper Ensemble ├─ ZK Node 2 (Follower)
Client 3 ──┘                    └─ ZK Node 3 (Follower)
```

- **Ensemble**: Cluster of ZooKeeper nodes (odd number: 3, 5, 7)
- **Quorum**: Majority must agree (3/5 nodes = tolerates 2 failures)
- **ZNodes**: Hierarchical data nodes (like a file system)

### Hadoop HA with ZooKeeper
```xml
<!-- hdfs-site.xml: NameNode HA configuration -->
<property>
    <name>dfs.ha.automatic-failover.enabled</name>
    <value>true</value>
</property>
<property>
    <name>ha.zookeeper.quorum</name>
    <value>zk1:2181,zk2:2181,zk3:2181</value>
</property>
```

### Services That Use ZooKeeper
- **HDFS HA**: NameNode failover
- **YARN HA**: ResourceManager failover
- **HBase**: Region server coordination
- **Kafka**: Broker management (legacy, now KRaft)
- **Hive**: Lock management

### Interview Tip
ZooKeeper solves the fundamental problem of **distributed consensus** — how multiple nodes agree on state. In Hadoop HA, it detects NameNode failure and triggers automatic failover to the standby. Key fact: ZooKeeper requires an **odd number** of nodes to form a quorum.

---

## Question 15

**How doesHadoop handlethefailure of a datanode?**

**Answer:**

### Detection Mechanism
The NameNode detects DataNode failures through **heartbeat signals**.

| Step | What Happens |
|------|--------------|
| 1. **Heartbeat timeout** | DataNode stops sending heartbeats (default: every 3 seconds) |
| 2. **Marked dead** | NameNode marks DataNode as dead after ~10 minutes (configurable) |
| 3. **Under-replicated blocks** | NameNode identifies blocks that lost a replica |
| 4. **Re-replication** | NameNode instructs other DataNodes to copy under-replicated blocks |
| 5. **Rack awareness** | New replicas placed according to rack-aware policy |

### Process Flow
```
DataNode X dies
    ↓
NameNode detects (no heartbeat for dfs.namenode.heartbeat.recheck-interval)
    ↓
NameNode scans block map for blocks stored on DataNode X
    ↓
Blocks with replicas < dfs.replication (default 3) are marked under-replicated
    ↓
NameNode schedules re-replication on healthy DataNodes
    ↓
Replication factor restored (transparent to clients)
```

### Configuration
```xml
<!-- hdfs-site.xml -->
<property>
    <name>dfs.heartbeat.interval</name>
    <value>3</value>  <!-- Heartbeat every 3 seconds -->
</property>
<property>
    <name>dfs.namenode.heartbeat.recheck-interval</name>
    <value>300000</value>  <!-- 5 minutes recheck -->
</property>
<property>
    <name>dfs.replication</name>
    <value>3</value>  <!-- Default replication factor -->
</property>
```

### Key Points
- **No data loss** as long as at least 1 replica survives
- **Rack awareness** ensures replicas are on different racks
- **Automatic recovery** — no manual intervention needed
- **Decommissioning** allows graceful removal of nodes

### Interview Tip
The NameNode doesn't immediately declare a DataNode dead after missing one heartbeat — it waits for `2 * heartbeat.recheck-interval + 10 * heartbeat.interval` (default ~10.5 minutes). This avoids false positives from network blips. Mention that this is why Hadoop favors **high replication** over single-copy storage.

---

## Question 16

**Explain the process ofdata replicationinHDFS.**

**Answer:**

### Definition
HDFS replicates each data block across multiple DataNodes to ensure **fault tolerance** and **data availability**.

### Replication Process

| Step | Action |
|------|--------|
| 1. **Client writes** | Client sends block to first DataNode |
| 2. **Pipeline replication** | First DN forwards to second, second to third |
| 3. **Acknowledgment** | ACKs flow back through pipeline |
| 4. **NameNode metadata** | NameNode records block locations |

### Pipeline Architecture
```
Client → DataNode 1 → DataNode 2 → DataNode 3
              ← ACK  ←  ACK   ← ACK
```

### Rack-Aware Replica Placement
```
Default policy (replication factor = 3):
- Replica 1: Same node as writer (or random node)
- Replica 2: Different rack (fault tolerance across racks)
- Replica 3: Same rack as Replica 2, different node (bandwidth optimization)
```

| Replica | Location | Reason |
|---------|----------|--------|
| **1st** | Local node/rack | Low latency write |
| **2nd** | Remote rack | Rack-level fault tolerance |
| **3rd** | Same rack as 2nd | Balance between safety and bandwidth |

### Configuration
```xml
<!-- hdfs-site.xml -->
<property>
    <name>dfs.replication</name>
    <value>3</value>  <!-- Default replication factor -->
</property>
<property>
    <name>dfs.replication.max</name>
    <value>512</value>  <!-- Maximum allowed -->
</property>
```

```bash
# Change replication for specific file
hdfs dfs -setrep -w 5 /path/to/important_file

# Check replication status
hdfs fsck /path/to/file -files -blocks -locations
```

### Interview Tip
The key design choice is the **pipeline replication** model — the client only sends data once, and DataNodes forward to each other. This minimizes client bandwidth usage. Also, the rack-aware placement policy balances **fault tolerance** (cross-rack) with **write performance** (intra-rack for later replicas).

---

## Question 17

**What isspeculative executioninHadoop, and why is itused?**

**Answer:**

### Definition
Speculative execution is a Hadoop optimization where the framework launches **duplicate copies of slow-running tasks** on other nodes, using the output of whichever finishes first.

### How It Works

| Step | Action |
|------|---------|
| 1. **Monitor** | YARN tracks task progress across all mappers/reducers |
| 2. **Detect straggler** | Task running significantly slower than average |
| 3. **Launch backup** | Start duplicate task on a different node |
| 4. **First wins** | Use output from whichever copy finishes first |
| 5. **Kill duplicate** | Terminate the slower copy |

```
Node A (slow):  [=====...........]  ← Straggler detected
Node B (backup): [============]     ← Backup launched, finishes first ✅
Node A:          [killed]           ← Original killed
```

### Why It's Needed
- **Hardware heterogeneity**: Some nodes are slower (disk issues, old hardware)
- **Resource contention**: Competing workloads slow down tasks
- **Data skew**: Some tasks process more data
- **Network issues**: Slow rack switch or congestion

### Configuration
```xml
<!-- mapred-site.xml -->
<property>
    <name>mapreduce.map.speculative</name>
    <value>true</value>  <!-- Default: true -->
</property>
<property>
    <name>mapreduce.reduce.speculative</name>
    <value>true</value>  <!-- Default: true -->
</property>
```

### When to Disable
- **Non-idempotent tasks**: Writing to external databases (duplicates!)
- **Resource-constrained clusters**: Backup tasks waste resources
- **Tasks with side effects**: Email sending, API calls

### Interview Tip
Speculative execution trades **extra resources** for **lower latency**. It works because Hadoop clusters often have idle capacity. Disable it for tasks with side effects or when cluster utilization is already high (>80%).

---

## Question 18

**What is the significance of theinput splitinMapReduce jobs?**

**Answer:**

### Definition
An **input split** is a logical division of input data that defines the chunk of data processed by a single mapper. It determines parallelism and data locality.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Block** | Physical storage unit in HDFS (default 128 MB) |
| **Input Split** | Logical division for MapReduce (usually = 1 block) |
| **Mapper** | One mapper per input split |
| **Data locality** | Split assigned to node storing the data |

### Split vs Block
```
File (512 MB) stored in HDFS:
  Block 1 (128 MB) → Split 1 → Mapper 1
  Block 2 (128 MB) → Split 2 → Mapper 2
  Block 3 (128 MB) → Split 3 → Mapper 3
  Block 4 (128 MB) → Split 4 → Mapper 4
```

### How Splits Are Created
```java
// InputFormat.getSplits() determines split strategy
// Default: FileInputFormat creates one split per block

// Custom split size
// mapreduce.input.fileinputformat.split.minsize = 256MB (fewer mappers)
// mapreduce.input.fileinputformat.split.maxsize = 64MB  (more mappers)

// Split size formula:
// splitSize = max(minSize, min(maxSize, blockSize))
```

### Configuration
```xml
<property>
    <name>mapreduce.input.fileinputformat.split.minsize</name>
    <value>0</value>  <!-- Default: 0 (use block size) -->
</property>
<property>
    <name>mapreduce.input.fileinputformat.split.maxsize</name>
    <value>268435456</value>  <!-- 256 MB -->
</property>
```

### Impact on Performance
- **Too many splits** → Too many mappers → Overhead from task startup
- **Too few splits** → Low parallelism → Underutilized cluster
- **Optimal**: Split size ≈ HDFS block size (default behavior)

### Interview Tip
The key insight is that splits are **logical** (defined by InputFormat) while blocks are **physical** (stored in HDFS). By default, 1 split = 1 block, which maximizes **data locality** — the mapper runs on the node storing the data, avoiding network transfer.

---

## Question 19

**How doespartitioningwork inHadoop, and when is itused?**

**Answer:**

### Definition
Partitioning determines **which reducer receives which key** during the shuffle phase. The default `HashPartitioner` distributes keys evenly across reducers.

### Partitioning Process
```
Mapper outputs (key, value) pairs
    ↓
Partitioner: partition = hash(key) % numReducers
    ↓
Each reducer gets all values for its assigned keys
```

### Default Partitioner
```java
public class HashPartitioner<K, V> extends Partitioner<K, V> {
    public int getPartition(K key, V value, int numReduceTasks) {
        return (key.hashCode() & Integer.MAX_VALUE) % numReduceTasks;
    }
}
```

### Custom Partitioner
```java
// Partition by country for geographic analysis
public class CountryPartitioner extends Partitioner<Text, IntWritable> {
    @Override
    public int getPartition(Text key, IntWritable value, int numReduceTasks) {
        String country = key.toString().split("_")[0];
        if (country.equals("US")) return 0;
        if (country.equals("EU")) return 1;
        return 2;  // Rest of world
    }
}

// Set in driver
job.setPartitionerClass(CountryPartitioner.class);
job.setNumReduceTasks(3);  // Must match partitioner logic
```

### When to Use Custom Partitioning

| Scenario | Reason |
|----------|--------|
| **Data skew** | Default hash creates uneven distribution |
| **Secondary sort** | Composite keys need custom partitioning |
| **Data locality** | Group related keys to same reducer |
| **Output organization** | Separate output files by category |
| **Total order sort** | Range-based partitioning |

### Interview Tip
Data skew is the biggest partitioning problem — one reducer gets most of the data while others sit idle. The solution is a **custom partitioner** that distributes hot keys across multiple reducers, or using a **combiner** to pre-aggregate before shuffling.

---

## Question 20

**Explain howreducerswork inMapReduceand their interaction withshufflers.**

**Answer:**

### Shuffle and Reduce Process

| Phase | What Happens |
|-------|--------------|
| 1. **Map output** | Mappers produce (key, value) pairs |
| 2. **Partition** | Partitioner assigns keys to reducers |
| 3. **Sort** | Map outputs sorted by key (on mapper side) |
| 4. **Shuffle** | Transfer sorted data from mappers to reducers over network |
| 5. **Merge sort** | Reducer merges sorted data from all mappers |
| 6. **Reduce** | Reducer processes all values for each key |

### Shuffle and Sort in Detail
```
Mapper 1: (A,1) (B,2) (A,3)     Mapper 2: (B,4) (A,5) (C,6)
    ↓ Sort                          ↓ Sort
    (A,1)(A,3)(B,2)                (A,5)(B,4)(C,6)
    ↓ Shuffle (network transfer)    ↓

Reducer 0 (keys A):     (A, [1, 3, 5])  ─→ reduce() → (A, 9)
Reducer 1 (keys B,C):   (B, [2, 4])     ─→ reduce() → (B, 6)
                         (C, [6])       ─→ reduce() → (C, 6)
```

### Reducer Mechanics
```java
public class SumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
    @Override
    public void reduce(Text key, Iterable<IntWritable> values, Context context) {
        int sum = 0;
        for (IntWritable val : values) {
            sum += val.get();  // Iterate through all values for this key
        }
        context.write(key, new IntWritable(sum));
    }
}
```

### Shuffle Optimization

| Optimization | Description |
|-------------|-------------|
| **Combiner** | Mini-reducer on mapper side (reduces shuffle data) |
| **Compression** | Compress map output before shuffle |
| **Sort buffer** | `mapreduce.task.io.sort.mb` (default 100 MB) |
| **Spill threshold** | `mapreduce.map.sort.spill.percent` (default 0.80) |

### Interview Tip
The shuffle phase is typically the **most expensive** part of MapReduce — it involves disk I/O (spilling), network transfer, and merge sorting. Using a **combiner** can reduce shuffle data by 10-100x. The combiner must be commutative and associative (e.g., sum, max, but not average).

---

## Question 21

**What areSequenceFilesinHadoop?**

**Answer:**

### Definition
SequenceFiles are Hadoop's **binary file format** that stores data as serialized key-value pairs. They are designed for efficient storage and processing within the Hadoop ecosystem.

### Compression Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Uncompressed** | No compression | Debugging, small files |
| **Record compressed** | Each value compressed independently | General purpose |
| **Block compressed** | Group of records compressed together | Best compression ratio |

### Structure
```
SequenceFile Format:
┌───────────────────────────┐
│ Header (version, key/val classes) │
├───────────────────────────┤
│ Record 1: (key1, value1)          │
│ Record 2: (key2, value2)          │
│ Sync Marker (every ~2000 records) │  ← Enables splitting
│ Record N: (keyN, valueN)          │
└───────────────────────────┘
```

### Code Example
```java
import org.apache.hadoop.io.*;
import org.apache.hadoop.conf.Configuration;

// Write SequenceFile
Configuration conf = new Configuration();
SequenceFile.Writer writer = SequenceFile.createWriter(conf,
    SequenceFile.Writer.file(new Path("/output/data.seq")),
    SequenceFile.Writer.keyClass(Text.class),
    SequenceFile.Writer.valueClass(IntWritable.class),
    SequenceFile.Writer.compression(CompressionType.BLOCK));

writer.append(new Text("key1"), new IntWritable(100));
writer.append(new Text("key2"), new IntWritable(200));
writer.close();

// Read SequenceFile
SequenceFile.Reader reader = new SequenceFile.Reader(conf,
    SequenceFile.Reader.file(new Path("/output/data.seq")));
Text key = new Text();
IntWritable val = new IntWritable();
while (reader.next(key, val)) {
    System.out.println(key + ": " + val);
}
```

### Advantages
- **Splittable** (sync markers enable MapReduce parallelism)
- **Binary format** (efficient serialization)
- **Compressible** (built-in compression support)
- **Small file solution** (merge many small files into one SequenceFile)

### Interview Tip
SequenceFiles solve the **small files problem** in HDFS — instead of storing millions of small files (each consuming a NameNode metadata entry), you merge them into SequenceFiles. This dramatically reduces NameNode memory pressure.

---

## Question 22

**Describe the ways tooptimizeaMapReduce job.**

**Answer:**

### Optimization Categories

| Category | Techniques |
|----------|------------|
| **Input** | Proper input format, split size tuning |
| **Map phase** | Combiner, in-mapper combining, compression |
| **Shuffle** | Compression, buffer tuning, partitioning |
| **Reduce** | Fewer reducers, secondary sort |
| **Output** | Compression, proper format |
| **Cluster** | JVM reuse, speculative execution |

### Key Optimizations
```xml
<!-- 1. Use Combiner (reduces shuffle data by 10-100x) -->
job.setCombinerClass(SumReducer.class);

<!-- 2. Compress map output (reduce shuffle traffic) -->
<property>
    <name>mapreduce.map.output.compress</name>
    <value>true</value>
</property>
<property>
    <name>mapreduce.map.output.compress.codec</name>
    <value>org.apache.hadoop.io.compress.SnappyCodec</value>
</property>

<!-- 3. Tune sort buffer (reduce disk spills) -->
<property>
    <name>mapreduce.task.io.sort.mb</name>
    <value>256</value>  <!-- Default: 100 MB -->
</property>
<property>
    <name>mapreduce.map.sort.spill.percent</name>
    <value>0.90</value>  <!-- Default: 0.80 -->
</property>

<!-- 4. JVM reuse (avoid JVM startup overhead) -->
<property>
    <name>mapreduce.job.jvm.numtasks</name>
    <value>-1</value>  <!-- Reuse JVM for all tasks -->
</property>

<!-- 5. Optimal number of reducers -->
job.setNumReduceTasks(cluster_nodes * reducers_per_node * 0.95);
```

### Advanced Optimizations
1. **Use efficient file formats**: ORC, Parquet (columnar, compressed)
2. **Avoid small files**: Merge with CombineFileInputFormat
3. **Data locality**: Ensure splits align with HDFS blocks
4. **Proper data types**: Use `Writable` types, not Java serialization
5. **Pre-sort data**: If doing joins, pre-sort by join key

### Interview Tip
The three highest-impact optimizations are: 1) **Combiner** (reduces network I/O), 2) **Map output compression** (Snappy for speed, Gzip for ratio), 3) **Proper number of reducers** (rule of thumb: 0.95 * total reduce slots). Always profile before optimizing — use Hadoop counters to find bottlenecks.

---

## Question 23

**What is the significance ofcombinerin theHadoop MapReduce framework?**

**Answer:** _[To be filled]_

---

## Question 24

**Explain what you can do tooptimize the performanceofHDFS.**

**Answer:** _[To be filled]_

---

## Question 25

**What are thebest practicesfor managingmemory and CPU resourcesin aHadoop cluster?**

**Answer:** _[To be filled]_

---

## Question 26

**What is the concept oferasure codinginHDFS, and how does it differ fromreplication?**

**Answer:** _[To be filled]_

---

## Question 27

**Explain howHadoop uses data localityto improveperformance.**

**Answer:** _[To be filled]_

---

## Question 28

**How doesHadoop support different file formats, and what are some of them?**

**Answer:** _[To be filled]_

---

## Question 29

**What isHadoop federation, and how can itscale a Hadoop cluster?**

**Answer:** _[To be filled]_

---

## Question 30

**What are the implications ofsmall filesonHDFS performanceand how can this bemitigated?**

**Answer:** _[To be filled]_

---
