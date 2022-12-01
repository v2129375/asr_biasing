import sys

class Node:
    value = 0
    action = ""
    last = ""
    def __init__(self,v) -> None:
        self.value = v

def min_distance(word1, word2):
 
    row = len(word1) + 1
    column = len(word2) + 1
    cache = []
    for i in range(row):
        cache.append([])
        for j in range(column):
            cache[i].append(Node(0))


    for i in range(row):
        for j in range(column):
 
            if i ==0 and j ==0:
                cache[i][j].value = 0
            elif i == 0 and j!=0:
                cache[i][j].value = j
            elif j == 0 and i!=0:
                cache[i][j].value = i
            else:
                if word1[i-1] == word2[j-1]:
                    cache[i][j].action = "correct"
                    cache[i][j].value = cache[i-1][j-1].value
                    cache[i][j].last = cache[i-1][j-1]
                else:
                    replace = cache[i-1][j-1].value + 1
                    insert = cache[i][j-1].value + 1
                    remove = cache[i-1][j].value + 1

                    this_min = sys.maxsize
                    this_action = ""
                    dic = {"replace":replace,"insert":insert,"remove":remove}
                    for k in dic.keys():
                        if dic[k] < this_min:
                            this_min = dic[k] 
                            this_action = k
                    if this_action == "replace":
                        cache[i][j].last = cache[i-1][j-1]
                    elif this_action == "insert":
                        cache[i][j].last = cache[i][j-1]
                    elif this_action == "remove":
                        cache[i][j].last = cache[i-1][j]

                    cache[i][j].action = this_action
                    cache[i][j].value = min(replace, insert, remove)
    # 回溯看看做了什么action
    action = []
    a = cache[row-1][column-1]
    while a.action != "":
        action.insert(0,a.action)
        a = a.last
    # print(action)
    wrong_index = []

    for i in range(len(action)):
        if action[i] == "insert":
            word1.insert(i," ")
        elif action[i] == "remove":
            word2.insert(i," ")
        elif action[i] == "replace":
            wrong_index.append(i)

    return word1,word2,wrong_index   
if __name__ == "__main__":
    list1 = ['他', '在', '落后', '的', '情况', '下', '打出', '九点', '踢', '环']
    list2 = ['她', '在', '落后', '的', '情况', '下', '打出', '九点', '七环']
    list1,list2,wrong_index = min_distance(list1, list2)
    print(list1)
    print(list2)
    print(wrong_index)