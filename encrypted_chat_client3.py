# chat_client.py

import hashlib
import multipiv3
import sys

print("Encrypted chat client v1")
alias_name=input("Alias name?")
if alias_name.strip()=="":
    alias_name="Me"
alias_name='{:^10}'.format(alias_name[0:10])

#e=multipiv3.AESCipher(str(hashlib.md5(sys.argv[3].encode('utf-8')).digest()))
e=multipiv3.AESCipher(str(hashlib.sha256(sys.argv[3].encode('utf-8')).digest()))

hasher=multipiv3.multipi()
hasher.chat_client_encrypted(e,alias_name)



