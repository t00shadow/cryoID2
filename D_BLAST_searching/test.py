class Solution:

    def totalFruit(self, fruits) -> int:
        nowFruit, maxFruit = 0, 0
        kind1, kind2, lastPos1, lastPos2 = 999999, 999999, 0, 0
        for i in range(len(fruits)):
            if fruits[i] == kind1:
                nowFruit += 1
                lastPos1 = i
            elif fruits[i] == kind2:
                nowFruit += 1
                lastPos2 = i
            else:
                maxFruit = max(nowFruit, maxFruit)
                if kind1 == 999999:
                    nowFruit = i + 1
                    lastPos1, kind1 = i, fruits[i]
                elif kind2 == 999999:
                    nowFruit = i + 1
                    lastPos2, kind2 = i, fruits[i]
                elif lastPos1 < lastPos2:
                    nowFruit = i - lastPos1
                    lastPos1, kind1 = i, fruits[i]
                elif lastPos2 < lastPos1:
                    nowFruit = i - lastPos2
                    lastPos2, kind2 = i, fruits[i]
        return max(nowFruit, maxFruit)


A = Solution()
a = A.totalFruit([0, 1, 6, 6, 4, 4, 6])
print(a)
