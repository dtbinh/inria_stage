##This script is to generate the names of different combination of algorithms, problems, dimensions
import os,sys

def write_config_file(file_text):
    file = open("/home/omohamme/INRIA/Software/folders/limbo_master/limbo/src/benchmarks/wscript","w")
    file.write(file_text)
    file.close()
    print "wscript creation is complete"

file_header = "#! /usr/bin/env python\n\
import limbo\n\
def build(bld):\n\
    limbo.create_variants(bld,\n\
    source = 'multi.cpp',\n\
    uselib_local = 'limbo',\n\
    uselib = 'BOOST EIGEN TBB SFERES',\n\
    variants = "

def compile_limbo():
    #os.system("./home/omohamme/INRIA/Software/folders/limbo_master/limbo/omar_lazy.sh")
    current_path = os.getcwd()
    os.chdir("/home/omohamme/INRIA/Software/folders/limbo_master/limbo/")
    
    os.system("rm -rf /home/omohamme/INRIA/Software/folders/limbo_master/limbo/build")
    os.system("rm -rf /home/omohamme/INRIA/Software/folders/limbo_master/limbo/src/benchmarks/multi_*.*")
    os.system("/home/omohamme/INRIA/Software/folders/limbo_master/limbo/waf configure --sferes /home/omohamme/sferes2/sferes2/")
    os.system("/home/omohamme/INRIA/Software/folders/limbo_master/limbo/waf build")
    
    os.chdir(current_path)
    
def gen_config (algorithms,dimensions,problems,iterations):
    sep = ""
    for i in range(4): #just to make the text looks nice
        sep += "\t"
        
    final = "["
    for alg in algorithms:
        for prob in problems:
            for dim in dimensions:
                for iter in iterations:
                    final += "'" + " ".join(map(lambda y:str(y),(alg,prob,dim,iter))) + "',\n"+sep
    final = final[:-2]+"],)"            
    write_config_file(file_header + final)
    compile_limbo()
    print "LIMBO has been compiled"
    
#gen_config (algorithms,dimensions,problems,iterations)
