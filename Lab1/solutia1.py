
import numpy as np

def parser_function():
    f = open("sistem.txt", "r")
    x = f.read().split("\n")
    m = [[0 for _ in range(3)] for _ in range(3)]
    mr = [[0 for _ in range(1)] for _ in range(3)]
    signal = 1
    result = -1
    line = 0
    xflag = 0
    yflag =0
    zflag = 0
    m = [[0 for _ in range(3)] for _ in range(3)]
    mr = [[0 for _ in range(1)] for _ in range(3)]
    for i in x:
        for j in i.split(" "):
            if j.find("-") != -1:
                signal = -1
            if j.find("+") != -1:
                signal = 1
            if j.find("x") != -1:
                xflag = 1
                if j.split("x")[0].find("-") == -1:
                    # print(int(j.split("x")[0]) * signal) 
                    m[line][0] = int(j.split("x")[0]) * signal
                if j.split("x")[0].find("-") != -1:
                    # print(int(j.split("x")[0].split("-")[1]) * signal)
                    m[line][0] = int(j.split("x")[0].split("-")[1]) * signal
            if j.find("y") != -1:
                yflag = 1
                if j.split("y")[0].find("-") == -1:
                    m[line][1] = int(j.split("y")[0]) * signal
                if j.split("y")[0].find("-") != -1:
                    m[line][1] = int(j.split("y")[0].split("-")[1]) * signal         
            if j.find("z") != -1:
                zflag = 1
                if j.split("z")[0].find("-") == -1:
                    m[line][2] = int(j.split("z")[0]) * signal
                if j.split("z")[0].find("-") != -1:
                    m[line][2] = int(j.split("z")[0].split("-")[1]) * signal  
            if result == 1:
                mr[line][0] = int(j)
                result = -1
            if j.find("=") != -1:
                result = 1    
        if xflag == 0:
                m[line][0] = 0
        if yflag == 0:
                m[line][1] = 0
        if zflag == 0:
                m[line][2] = 0
        xflag = yflag = zflag = 0        
        line = line + 1            
    return m, mr



def calc_det(m):
    return m[0][0]*m[1][1]*m[2][2] + m[0][2]*m[1][0]*m[2][1] + m[0][1]*m[1][2]*m[2][0] - m[0][2]*m[1][1]*m[2][0] - m[0][1]*m[1][0]*m[2][2] - m[0][0]*m[1][2]*m[2][1]

def make_mt(m):
    mt =[[0 for _ in range(3)] for _ in range(3)]
    mt[0][0] = m[0][0]
    mt[1][1] = m[1][1]
    mt[2][2] = m[2][2]
    mt[0][1] = m[1][0]
    mt[0][2] = m[2][0]
    mt[1][0] = m[0][1]
    mt[1][2] = m[2][1]
    mt[2][0] = m[0][2]
    mt[2][1] = m[1][2] 
    return mt      

def make_mstar(mt,m):
    mstar = [[0 for _ in range(3)] for _ in range(3)]
    mstar[0][0] = m[1][1]*m[2][2] - m[1][2]*m[2][1]
    mstar[0][1] = -1 * (m[0][1] * m[2][2] - m[2][1]*m[0][2])
    mstar[0][2] = m[0][1]*m[1][2] - m[1][1]*m[0][2]
    mstar[1][0] = -1 * (m[1][0] * m[2][2] - m[1][2]*m[2][0])
    mstar[1][1] = m[0][0]*m[2][2] - m[0][2]*m[2][0]
    mstar[1][2] = -1* (m[0][0] * m[1][2] - m[0][2]* m[1][0])
    mstar[2][0] = m[1][0]*m[2][1] - m[1][1] *m[2][0]
    mstar[2][1] = -1 * (m[0][0] * m[2][1] - m[0][1]*m[2][0])
    mstar[2][2] = m[0][0] * m[1][1] - m[1][0]* m[0][1]  

    return mstar

def make_mi(m):
    mi = [[0 for _ in range(3)] for _ in range(3)]
    det = calc_det(m)
    mi = make_mstar(make_mt(m),m)
    for i in range (0,3):
        for j in range (0,3):
            mi[i][j] = mi[i][j] / det
           
    return mi

def get_solution():
    m = [[0 for _ in range(3)] for _ in range(3)]
    s = [[0 for _ in range(1)] for _ in range(3)]
    mr = [[0 for _ in range(1)] for _ in range(3)]
    m, mr = parser_function()
    mi = [[0 for _ in range(3)] for _ in range(3)]
    det = calc_det(m)
    if det == 0 :
        print("Sistemul are determinatul 0")
    mi = make_mi(m)
    s[0][0] = mi[0][0]*mr[0][0] + mi[0][1]*mr[1][0] + mi[0][2]*mr[2][0]
    s[1][0] = mi[1][0]*mr[0][0] + mi[1][1]*mr[1][0]+ mi[1][2]*mr[2][0]
    s[2][0] = mi[2][0]*mr[0][0] + mi[2][1]*mr[1][0]+ mi[2][2]*mr[2][0]
    print(s)


def get_solution_np():
    m = [[0 for _ in range(3)] for _ in range(3)]
    mi = [[0 for _ in range(3)] for _ in range(3)]
    mr = [[0 for _ in range(1)] for _ in range(3)]
    m, mr = parser_function()
    mi = np.linalg.inv(m)
    s =  np.dot(mi,mr)
    print(s)    


print("Rezultat parsare")
print(parser_function())

print("solutie cu np")
get_solution_np()
print("solutie fara np")
get_solution()