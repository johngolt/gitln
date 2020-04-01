class cache:
    def __init__(self, func):
        self.func = func
        self.data = dict()

    def __call__(self, *args, **kwargs):
        key = '{}-{}-{}'.format(self.func.__name__, str(args), str(kwargs))
        if key in self.data:
            result = self.data.get(key)
        else:
            result = self.func(*args, **kwargs)
            self.data[key] = result
        return result


@cache
def area(length, width):
    return length * width
