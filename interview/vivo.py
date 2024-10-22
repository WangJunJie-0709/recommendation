# 反转链表
def reverse_list(node):
    if node is None:
        return node
    pre = None
    cur = node
    while cur:
        tmp = cur.next
        cur.next = pre
        pre = cur
        cur = tmp
    return pre