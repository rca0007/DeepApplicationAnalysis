'''
Created on Jan 30, 2018

@author: rober
'''

import socket

def main():
    
    sysdigListener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sysdigListener.bind(('', 10000))
    sysdigListener.listen(1)
    conn, addr = sysdigListener.accept()
    
    print('Connected to', addr)
    
    while True:
        data = conn.recv(4096)
        print(data)
    
    
if __name__ == "__main__": main()