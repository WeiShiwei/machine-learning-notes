

def isPalindrome(s):
    s = s.strip()
    if s == '':
        return True
    
    s = s.lower()
    i,j = 0,len(s)-1
    while i<j:
        if not s[i].isalnum():
            i = i+1
            continue
        
        if not s[j].isalnum():
            j = j-1
            continue

        if s[i] == s[j]:
            i = i+1
            j = j-1
        else:
            print(i,j, s[i],s[j])
            return False
        
    return True


s = "A man, a plan, a canal: Panama"
print(isPalindrome(s))
s1 = "race a car"
print(isPalindrome(s1))