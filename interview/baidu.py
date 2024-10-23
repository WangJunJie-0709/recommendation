def MaxLength(nums):
   if nums == []:
      return 0
   quene = []
   n = len(nums)
   for i in range(n):
      if quene == [] or nums[i] > quene[-1]:
         quene.append(nums[i])
      else:
         l, r = 0, len(quene) - 1
         while l < r:
            mid = (l + r) // 2
            if quene[mid] > nums[i]:
               r = mid - 1
            elif quene[mid] <= nums[i]:
               l = mid + 1
         quene[r] = nums[i]
   return len(quene)


if __name__ == '__main__':
   candi = [[10,9,2,5,3,7,101,18],[0,1,0,3,2,3],[7,7,7,7,7,7,7]]
   for nums in candi:
      print(MaxLength(nums))

