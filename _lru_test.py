class _Node:
    __slots__ = ('key', 'value', 'prev', 'next')
    def __init__(self, key=None, value=None):
        self.key = key; self.value = value
        self.prev = None; self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._map = {}
        self._head = _Node(); self._tail = _Node()
        self._head.next = self._tail; self._tail.prev = self._head
    def _remove(self, node):
        node.prev.next = node.next; node.next.prev = node.prev
    def _add_front(self, node):
        node.next = self._head.next; node.prev = self._head
        self._head.next.prev = node; self._head.next = node
    def get(self, key):
        node = self._map.get(key)
        if node is None: return -1
        self._remove(node); self._add_front(node)
        return node.value
    def put(self, key, value):
        node = self._map.get(key)
        if node is not None:
            node.value = value; self._remove(node); self._add_front(node); return
        if len(self._map) >= self.capacity:
            lru = self._tail.prev; self._remove(lru); del self._map[lru.key]
        node = _Node(key, value); self._map[key] = node; self._add_front(node)

cache = LRUCache(2)
cache.put(1, 1); cache.put(2, 2)
assert cache.get(1) == 1
cache.put(3, 3)
assert cache.get(2) == -1
cache.put(4, 4)
assert cache.get(1) == -1
assert cache.get(3) == 3
assert cache.get(4) == 4
print("ALL PASS")

# 额外边界测试
c = LRUCache(1)
c.put(1, 1); c.put(2, 2)
assert c.get(1) == -1 and c.get(2) == 2
c.put(2, 22)
assert c.get(2) == 22
print("EDGE PASS")
