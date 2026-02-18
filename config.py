database = 'tc_cadet_dataset_db'

host = '/var/run/postgresql/'
# host = None

# Database user
user = 'postgres'

# The password to the database user
password = 'postgres'

# The port number for Postgres
port = '5432'


include_edge_type=[
    "EVENT_WRITE",
    "EVENT_READ",
    "EVENT_CLOSE",
    "EVENT_OPEN",
    "EVENT_EXECUTE",
    "EVENT_SENDTO",
    "EVENT_RECVFROM",
]

rel2id = {
 1: 'EVENT_WRITE',
 'EVENT_WRITE': 1,
 2: 'EVENT_READ',
 'EVENT_READ': 2,
 3: 'EVENT_CLOSE',
 'EVENT_CLOSE': 3,
 4: 'EVENT_OPEN',
 'EVENT_OPEN': 4,
 5: 'EVENT_EXECUTE',
 'EVENT_EXECUTE': 5,
 6: 'EVENT_SENDTO',
 'EVENT_SENDTO': 6,
 7: 'EVENT_RECVFROM',
 'EVENT_RECVFROM': 7
}

node_type_map = {
    'subject': 0,
    'file': 1,
    'netflow': 2
}

# ── Feature encoding constants ──
NUM_NODE_TYPES = 3          # file=0, netflow=1, subject=2
NUM_EDGE_TYPES = 7          # 7 operations (1-indexed in rel2id)
NODE_EMB_DIM = 384          # SentenceTransformer all-MiniLM-L6-v2 output dim
MSG_DIM = NODE_EMB_DIM + NUM_EDGE_TYPES + NODE_EMB_DIM   # 384+7+384 = 775
node_type_map = {'subject': 0, 'file': 1, 'netflow': 2}

# The directory to save all artifacts
artifact_dir = "./artifact/"

# The directory to save the vectorized graphs
graphs_dir = artifact_dir + "graphs/"

# The directory to save the models
models_dir = artifact_dir + "models/"

# Data settings
benign_ratio = 3

# Training settings

epochs = 100

train_days = [6,10]
test_days = [11,12]