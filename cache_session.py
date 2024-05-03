

class CacheSession:
    def __init__(self):
        self.caches = {}

    def new_cache(self, cache_id, obj):
        assert cache_id not in self.caches, f'Cache {cache_id} already exists'
        self.caches[cache_id] = obj

    def get_cache(self, cache_id):
        assert cache_id in self.caches, f'Cache {cache_id} does exists'
        return self.caches[cache_id]

    def del_cache(self, cache_id):
        assert cache_id in self.caches, f'Cache {cache_id} does not exist'
        del self.caches[cache_id]

    def __contains__(self, cache_id):
        return cache_id in self.caches


