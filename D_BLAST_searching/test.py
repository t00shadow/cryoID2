class Solution:
    def minWindow(self, s: str, t: str) -> str:
        from collections import Counter, defaultdict
        sCounts = defaultdict(int)
        tCounts = defaultdict(int)
        minLen = float("inf")
        res = ""
        for char, count in Counter(s).items():
            sCounts[char] = count
        for char, count in Counter(t).items():
            tCounts[char] = count
        for key, value in tCounts.items():
            if sCounts[key] < value:
                return res

        countsLeft = tCounts
        start, end = 0, 0
        charLeft = len(t)
        while end < len(s):
            if countsLeft[s[end]] > 0:
                charLeft -= 1
            countsLeft[s[end]] -= 1
            end += 1
            while charLeft == 0:
                if end - start < minLen:
                    minLen = end - start
                    res = s[start:end]
                if countsLeft[s[start]] == 0:
                    charLeft += 1
                countsLeft[s[start]] += 1
                start += 1
        return res


s = "ADOBECODEBANC"
t = "ABC"


A = Solution()
B = A.minWindow(s, t)
print(B)


# while end < len(s):
#     if lookup[s[end]] > 0:
#         counter -= 1
#     lookup[s[end]] -= 1
#     end += 1
#     while counter == 0:
#         if min_len > end - start:
#             min_len = end - start
#             res = s[start:end]
#         if lookup[s[start]] == 0:
#             counter += 1
#         lookup[s[start]] += 1
#         start += 1
# return res
