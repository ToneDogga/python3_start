#import base64
import hashlib
import multipiv3
import sys

print("Encrypted chat server v1")
pp=input("Passphrase?")
alias_name=input("Alias name? (10 chars max):")
if alias_name.strip()=="":
    alias_name="Me"
alias_name='{:^10}'.format(alias_name[0:10])

#e=multipiv3.AESCipher(str(hashlib.md5(pp.encode('utf-8')).digest()))
e=multipiv3.AESCipher(str(hashlib.sha256(pp.encode('utf-8')).digest()))

hasher=multipiv3.multipi()
hasher.chat_server_encrypted(e,alias_name)
