class PrefetchBufferHandler:

    def __init__(self, fn):
        self.fn = fn
        self.buffer = {}

    def __call__(self, *args):
        print(f'__call__ with args: {args}')
        # if args not in self.buffer:
        #     self.buffer[args] = self.fn(*args)
        # return self.buffer[args]
        return self.fn(*args)