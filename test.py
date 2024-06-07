import asyncio
import time
from typing import List
import os
import aiotieba as tb
from aiotieba import PostSortType
from aiotieba.logging import get_logger as LOG
import pandas as pd

BDUSS = "dETjN3TjVuT3MwdzJhTjV0V3dtbE92ckVZdWJ6Y1V4emxtYUMtVzNGcER5SDltSUFBQUFBJCQAAAAAAAAAAAEAAADI75CkzfVKSm5pY2UAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEM7WGZDO1hmSz"
thread_list = []
reply_list = []

meaning_less = ['出院', '弱智吧']


async def crawler(fname: str):

    start_time = time.perf_counter()
    LOG().info("Spider start")


    # 使用键名"default"对应的BDUSS创建客户端
    async with tb.Client(BDUSS) as client:
        # asyncio.Queue是一个任务队列
        # maxsize=8意味着缓冲区长度为8
        # 当缓冲区被填满时，调用Queue.put的协程会被阻塞
        task_queue = asyncio.Queue(maxsize=8)
        # 当is_running被设为False后，消费者会在超时后退出
        is_running = True

        async def producer():
            """
            生产者协程
            """

            for pn in range(10000000, 0, -1):
                # 生产者使用Queue.put不断地将页码pn填入任务队列task_queue
                await task_queue.put(pn)
            # 这里需要nonlocal来允许对闭包外的变量的修改操作（类似于引用传递和值传递的区别）
            nonlocal is_running
            # 将is_running设置为False以允许各消费协程超时退出
            is_running = False

        async def worker(i: int):
            """
            消费者协程

            Args:
                i (int): 协程编号
            """

            while 1:
                try:
                    # 消费者协程不断地使用Queue.get从task_queue中拉取由生产者协程提供的页码pn作为任务
                    # asyncio.wait_for会等待作为参数的协程执行完毕直到超时
                    # timeout=1即把超时时间设为1秒
                    # 如果超过1秒未获取到新的页码pn，asyncio.wait_for(...)将抛出asyncio.TimeoutError
                    pn = await asyncio.wait_for(task_queue.get(), timeout=1)
                    LOG().debug(f"Worker#{i} handling pn:{pn}")
                except asyncio.TimeoutError:
                    # 捕获asyncio.TimeoutError以退出协程
                    if is_running is False:
                        # 如果is_running为False，意味着不需要再轮询task_queue获取新任务
                        LOG().debug(f"Worker#{i} quit")
                        # 消费者协程通过return退出
                        return
                else:
                    # 执行被分派的任务，即爬取pn页的帖子列表
                    threads = await client.get_threads(fname, pn)
                    skip_thread = False
                    for thread in threads:
                        if skip_thread:
                            skip_thread = False

                        if thread.user.level < 4 or thread.contents.objs == [] or not isinstance(thread.contents.objs[0], tb.api._classdef.contents.FragText):
                            # 用户等级大于等于4, 且帖子标题为文本类型
                            continue

                        title = thread.title if thread.title != '' else thread.contents.objs[0].text
                        replys = await client.get_posts(thread.tid, rn=100)
                        for word in meaning_less:  # 过滤包含无意义词的帖子
                            if word in title:
                                skip_thread = True
                                break

                        if skip_thread:
                            continue  # 跳过当前thread的剩余处理
                        tmp = []

                        for reply in replys:
                            if reply.contents.objs == [] or not isinstance(reply.contents.objs[0], tb.api._classdef.contents.FragText):
                                # 回复类型需要为文本类型
                                continue
                            if reply.user.user_id == thread.user.user_id and title != reply.contents.objs[0].text:
                                # 如果回复和标题的用户是同一个人, 将其拼接
                                title = title + '。' + reply.contents.objs[0].text
                                continue
                            reply_num = reply.reply_num
                            disagree = reply.disagree
                            agree = reply.agree
                            answer = reply.contents.objs[0].text
                            if answer in meaning_less or len(answer) == 1 or answer == title:
                                # 过滤无意义回复
                                continue

                            print("The title is " + f"{title}" " and the reply is " + f"{answer}")
                            tmp.append([answer, agree, disagree, reply_num])
                        thread_list.append(title)
                        reply_list.append(tmp)
                    # 这里的nonlocal同样是为了修改闭包外的变量thread_list
                    # nonlocal thread_list
                    # nonlocal reply_list

        # 创建消费者协程
        workers = [worker(i) for i in range(512)]

        # 使用asyncio.gather并发执行
        # 需要注意这里*workers中的*意为将列表展开成多个参数
        # 因为asyncio.gather只接受协程作为参数，不接受协程列表
        await asyncio.gather(*workers, producer())


    LOG().info(f"Spider complete. Time cost: {time.perf_counter()-start_time:.4f} secs")

    # 筛选出优质回复
    for reply in reply_list:
        reply[:] = sorted(reply, key=lambda x: (-x[1], x[2], -x[3]))[:min(5, len(reply))]

    paired_data = [
        (thread, reply[0], reply[1], reply[2], reply[3])  # 分别对应title, reply, agree, disagree, reply_num
        for thread, replies in zip(thread_list, reply_list)
        for reply in replies
        if reply[1] >= 10
    ]

    df = pd.DataFrame(paired_data, columns=['title', 'Reply', 'agree', 'disagree', 'reply_num'])

    df = df.drop_duplicates()
    csv_file_path = 'ruozhiba.csv'

    if os.path.exists(csv_file_path):
        os.remove(csv_file_path)

    df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')

# 执行协程crawler
asyncio.run(crawler("弱智"))
