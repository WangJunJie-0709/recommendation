# 给定一个非负整数数组 nums，你最初位于数组的第一个位置。数组中的每个元素代表你在该位置可以跳跃的最大长度。判断你是否能够到达数组的最后一个位置。
#
# 示例 1:
#
# 输入: nums = [2,3,1,1,4]
# 输出: true
# 解释: 我们可以从位置 0 跳到 1，跳 1 步，然后跳 3 步到达最后一个位置。
# 示例 2:
#
# 输入: nums = [3,2,1,0,4]
# 输出: true
# 解释: 我们可以从位置 0 跳到 1，跳 2 步，到达位置 3，然后跳 4 步到达最后一个位置。
# 示例 3:
#
# 输入: nums = [3,0,2,1,2]
# 输出: false
# 解释: 你将被困在初始位置，因为从第一个位置你不能跳到任何其他位置。


def jump(nums):
    n = len(nums)
    Next = 0
    i = 0
    while i <= Next:
        Next = max(Next, i + nums[i])
        if Next >= n - 1:
            return True
        i += 1
    return False

if __name__ == '__main__':
    candi = [[2,3,1,1,4], [3,2,1,0,4], [3,0,2,1,2]]
    for nums in candi:
        print(jump(nums))
