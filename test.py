import os
from urllib import request
import random

def saveSimpleImg(path):
    if not os.path.exists('source_img'):
        os.mkdir('source_img')

    pwd = "0123456789QWERTYUIOPASDFGHJKLZXCVBNM"
    for i in range(1000):
        code = ""
        for j in range(4):
            x = random.randint(0, len(pwd)-1)
            code += pwd[x]
        u = request.urlopen(path + code)
        data = u.read()
        imgPath = "source_img/%d.png" % (i)
        with open(imgPath, 'wb') as f:
            f.write(data)

saveSimpleImg("http://localhost:8080/login_j2ee/refpic.jsp?pwdcode=")