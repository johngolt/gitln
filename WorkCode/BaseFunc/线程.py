import concurrent.futures as futures
import requests
import threading
import time
import asyncio
import aiohttp
import multiprocessing
import random

'''线程本地存储。Threading.local()会创建一个对象，
它看起来像一个全局对象但又是特定于每个线程的。'''


thread_local = threading.local()


def get_session():
    if not getattr(thread_local, 'session', None):
        thread_local.session = requests.Session()
    return thread_local.session


def download_site(url):
    session = get_session()
    with session.get(url) as response:
        print(f'read{len(response.content)} from {url}')


def download_all_sites(sites):
    with futures.ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(download_site, sites)


if __name__ == '__main__':
    sites = ['https://www.jython.org',
    'http://olympus.realpython.org/dice']*80
    start_time = time.time()
    download_all_sites(sites)
    duration = time.time()-start_time
    print(f'Download{len(sites)} in {duration} seconds')


async def download_site1(session, url):
    async with session.get(url) as response:
        print('Read {} from {}'.format(response.content_length, url))


'''这些任务可以共享会话，因为它们都在同一个线程上运行。当会话处于糟糕（bad）状态时，一个任务不可能中断另一个任务。

在该上下文管理器中，它使用asyncio.ensure_future()
创建一个任务列表，该列表还负责启动这些任务。所有任务创建之后，
这个函数会使用asyncio.gather()
保持会话上下文处于活动状态，直到所有任务都完成为止。'''


async def download_all_sites1(sites):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in sites:
            task = asyncio.ensure_future(download_site1, url)
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)

asyncio.get_event_loop().run_until_complete(download_all_sites1(sites))


session = None


def get_global_session():
    global session
    if not session:
        session = requests.Session()


def download_site2(url):
    with session.get(url) as response:
        name = multiprocessing.current_process().name
        print(f'{name}: Read{len(response.content)} from {url}')


def downlad_all_sites(sites):
    with multiprocessing.Pool(initializer=get_global_session) as pool:
        pool.map(download_site2, sites)


SENTINEL = object()


class Pipeline:
    '''
    class to allow a single element pipeline between producer and consumer
    '''
    def __init__(self):
        self.message = 0
        self.producer_lock = threading.Lock()
        self.consumer_lock = threading.Lock()
        self.consumer_lock.acquire()
        '''允许生产者添加新的消息，但是消费者需要等待消息出现。'''

    def get_message(self):
        self.consumer_lock.acquire()
        message = self.message
        self.producer_lock.release()
        return message

    def set_message(self, message):
        self.producer_lock.acquire()
        self.message = message
        self.consumer_lock.release()


def producer(pipeline):
    for index in range(10):
        message = random.randint(1, 101)
        pipeline.set_message(message)
    pipeline.set_message(SENTINEL)


def consumer(pipeline):
    message = 0
    while message is not SENTINEL:
        message = pipeline.get_message()


if __name__ == '__main__':
    pipeline = Pipeline()
    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(producer, pipeline)
        executor.submit(consumer, pipeline)


class Pipeline2(queue.Queue):
    def __init__(self):
        super().__init__(maxsize=10)

    def get_message(self):
        value = self.get()
        return value

    def set_message(self, value):
        self.put(value)


def producer2(pipeline, event):
    while not event.is_set():
        message = random.randint(1, 101)
        pipeline.set_message(message)


def consumer2(pipeline, event):
    while not event.is_set() or not pipeline.empty():
        message = pipeline.get_message()


if __name__ == "__main__":
    pipeline = Pipeline2()
    event = threading.Event()
    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(producer, pipeline, event)
        executor.submit(consumer, pipeline, event)
        time.sleep(0.1)
        event.set()
