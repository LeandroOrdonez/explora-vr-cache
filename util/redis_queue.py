import redis

class RedisQueue(object):
    """Simple Queue with Redis Backend"""
    def __init__(self, name, namespace='queue', **redis_kwargs):
        """The default connection parameters are: host='localhost', port=6379, db=0"""
        self.__db= redis.Redis(**redis_kwargs)
        self.lkey = '%s:%s_list' %(namespace, name)
        self.skey = '%s:%s_set' %(namespace, name)

    def qsize(self):
        """Return the approximate size of the queue."""
        return self.__db.scard(self.skey)

    def empty(self):
        """Return True if the queue is empty, False otherwise."""
        return self.qsize() == 0

    def put(self, item):
        """Put item into the queue."""
        with self.__db.pipeline() as pipe:
            while 1:
                try:
                    pipe.watch(self.lkey)
                    pipe.watch(self.skey)
                    pipe.multi()
                    pipe.rpush(self.lkey, item)
                    pipe.sadd(self.skey, item)
                    pipe.execute()
                    break
                except redis.WatchError:
                    print(f"[RedisQueue] WatchError {self.lkey}, {self.skey}; retrying")

    def get(self, block=True, timeout=None):
        """Remove and return an item from the queue. 

        If optional args block is true and timeout is None (the default), block
        if necessary until an item is available."""
        with self.__db.pipeline() as pipe:
            while 1:
                try:
                    pipe.watch(self.skey)
                    if block:
                        item = pipe.blpop(self.lkey, timeout=timeout)
                    else:
                        item = pipe.lpop(self.lkey)
                    if item:
                        item = item[1]
                    pipe.multi()
                    pipe.srem(self.skey, item.decode('utf-8'))
                    pipe.execute()
                    break
                except redis.WatchError:
                    print(f"[RedisQueue] WatchError {self.skey}; retrying")
            return item
    
    def contains(self, item):
        """Checks if item exists in the queue."""
        return self.__db.sismember(self.skey, item)

    def get_nowait(self):
        """Equivalent to get(False)."""
        return self.get(False)
