
import hashlib, os

def read_chunks(file_handle, chunk_size=8192):
    while True:
        data = file_handle.read(chunk_size)
        if not data:
            break
        yield data

def sha256(file_handle):
    hasher = hashlib.sha256()
    for chunk in read_chunks(file_handle):
        hasher.update(chunk)
    return hasher.hexdigest()


def split_a_file_in_2(infile):

        #infile = open("input","r")

        with open(infile,'r') as f:
            linecount= sum(1 for row in f)

        splitpoint=linecount/2

        f.close()

        infilename=os.path.splitext(infile)[0]

        f = open(infile,"r")
        outfile1 = open(infilename+"001.csv","w")
        outfile2 = open(infilename+"002.csv","w")

        print("linecount=",linecount , "splitpoint=",splitpoint)

        linecount=0

        for line in f:
            linecount=linecount+1
            if ( linecount <= splitpoint ):
                outfile1.write(line)
            else:
                outfile2.write(line)

        f.close()
        outfile1.close()
        outfile2.close()


    
def count_file_rows(filename):
        with open(filename,'r') as f:
            return sum(1 for row in f)

   

def join2files_dos(in1,in2,out):
        os.system("copy /b "+in1+"+"+in2+" "+out)

def join2files_deb(in1,in2,out):
        os.system("cat "+in1+" "+in2+" "+out)



try:
    with open("salestrans060719.csv", 'rb') as f:
        hash_string = sha256(f)
    print("hash=",hash_string)
except IOError as e:
    print("error test")


print(count_file_rows("salestrans060719.csv"))

split_a_file_in_2("salestrans060719.csv")


join2files_deb("salestrans060719001.csv","salestrans060719002.csv","newsalestrans060719.csv")
print(count_file_rows("newsalestrans060719.csv"))

try:
    with open("newsalestrans060719.csv", 'rb') as f:
        hash_string = sha256(f)
    print("hash=",hash_string)
except IOError as e:
    print("error test")


