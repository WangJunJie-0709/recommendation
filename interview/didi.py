# 给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。
# 输入: s = "abcabcbb"
# 输出: 3
# 解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。


from collections import defaultdict

def maxSubList(s):
    n = len(s)
    if n <= 1:
        return n
    res = 0
    l, r = 0, 0
    dic = defaultdict(int)
    while r < n:
        ch = s[r]
        dic[ch] += 1
        while not helper(dic):
            border = s[l]
            dic[border] -= 1
            l += 1
        res = max(res, r - l + 1)
        r += 1
    return res


def helper(dic):
    for key in dic:
        if dic[key] > 1:
            return False
    return True


if __name__ == '__main__':
    s = "abcabcbb"
    print(maxSubList(s))
