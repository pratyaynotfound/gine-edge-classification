import time
from time import mktime
from datetime import datetime
import pytz
import psycopg2

from config import *

def datetime_to_ns_time_US(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    tz = pytz.timezone('US/Eastern')
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    dt = datetime.fromtimestamp(mktime(timeArray))
    timestamp = tz.localize(dt)
    timestamp = timestamp.timestamp()
    timeStamp = timestamp * 1000000000
    return int(timeStamp)

def init_database_connection():
    if host is not None:
        connect = psycopg2.connect(database = database,
                                   host = host,
                                   user = user,
                                   password = password,
                                   port = port
                                  )
    else:
        connect = psycopg2.connect(database = database,
                                   user = user,
                                   password = password,
                                   port = port
                                  )
    cur = connect.cursor()
    return cur, connect

def path2higlist(p):
    """Hierarchical path tokenization."""
    l = []
    p = (p or "").strip()
    if not p:
        return ["null"]
    is_abs = p.startswith("/")
    parts = [seg for seg in p.split("/") if seg]
    for seg in parts:
        if not l:
            l.append("/" + seg if is_abs else seg)
        else:
            l.append(l[-1] + "/" + seg)
    return l


def ip2higlist(p):
    """Hierarchical IP tokenization."""
    l = []
    spl = p.strip().split('.')
    for i in spl:
        if len(l) != 0:
            l.append(l[-1] + '.' + i)
        else:
            l.append(i)
    return l

def gen_nodeid2msg(cur):
    sql = "select * from node2id ORDER BY index_id;"
    cur.execute(sql)
    rows = cur.fetchall()
    nodeid2msg = {}
    for i in rows:
        nodeid2msg[i[0]] = i[-1]
        nodeid2msg[i[-1]] = {i[1]: i[2]}

    return nodeid2msg

def list2str(l):
    s=''
    for i in l:
        s += i + ' '
    return s