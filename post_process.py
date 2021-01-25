import recognition
import cv2

result_list = recognition.get_list()

swap = {'0': 'O', '1':'I','4':'A','5':'S', '6':'G', '7':'T', '8':'B'}
alphabet = 'QWERTYUIOPASDFGHJKLZXCVBNM'

# Adjust sequence 
def count_alphabet(string): 
    count = 0
    for charac in string:
        if charac in alphabet:
            count += 1
    return count

result = []
for string in result_list:
    if count_alphabet(string)/len(string) >= 0.75 and len(string)<6:
        result.append(string)
        result_list.remove(string)
        
for string in result_list:        
    if len(string)>= 6:
        result.append(string)
        result_list.remove(string)
    
result.append(result_list[0])  

def alph2digi(string):
    for i in string:
        for x in swap:
            if i == swap[x]:
                string = string.replace(i,x)
                
    return string

def digi2alph(string):
    for i in string:
        if i in swap:
            string = string.replace(i,swap[i])
            
    return string

def check_result():
    prefix = result[0]
    numbers = result[1]
    iso_code = result[2]

    #prefix
    prefix = digi2alph(prefix[:3])
    prefix += 'U'

    #numbers
    numbers = numbers[:6]
    numbers = alph2digi(numbers)

    #iso_code
    iso_code = alph2digi(iso_code[:2])
    iso_code += 'G1'

    return prefix + ' ' +numbers+' '+iso_code

print(check_result())

 