#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os, subprocess, shlex

if len(sys.argv) < 2:
  print "USAGE: " + sys.argv[0] + " FILE"
  print "FILE contains one sentence per line"
  exit()

sentences_file = sys.argv[1]
tmpfolder = sentences_file + ".tmp/"
if not os.path.exists(tmpfolder):
    os.makedirs(tmpfolder)
sentences_file_modified = tmpfolder + "modified"
tokenized = tmpfolder + "tokenized"
lemmatized = tmpfolder + "lemmatized"
parsed = tmpfolder + "parsed.conll09"
parsed_conll06 = tmpfolder + "parsed.conll06"
collapsed_path = tmpfolder + "collapsed"
collapsed = collapsed_path + "/parsed.conll" # note: JobImText creates this file, explicitly naming it is impossible

path_to_Mate_tools = "external/Mate\ Tools/"
path_to_JobImText = "external/collapsing-asl/"
path_to_PropsDE = "external/PropsDE/props-de/"
collapsing_rules_file = path_to_PropsDE + "ext/resources/german_modified.txt"

stderr_file = tmpfolder + "stderr"

###################
import codecs

source = codecs.open(sentences_file, encoding='utf-8')
target = codecs.open(sentences_file_modified, encoding='utf-8', mode = "w")
for line in source:
    target.write(line.replace(u"„", " "))
source.close()
target.close()


###################
# tokenize : single line for each token, including punctuation
#java -cp transition-1.30.jar is2.util.Split testrun/germantesttext.txt > testrun/one-word-per-line.txt
print "INFO: tokenize"
cmd = "java -cp " + path_to_Mate_tools + "transition-1.30.jar is2.util.Split "
cmd += sentences_file_modified
with open(tokenized, "w") as tmp:
  with open(stderr_file, "a") as stderrfile:
    process = subprocess.Popen(shlex.split(cmd), stdout = tmp, stderr=stderrfile)
    (stdoutdata, stderrdata) = process.communicate()

###################
# lemmatize
print "INFO: lemmatize"
model_file = "lemma-ger-3.6.model"
cmd = "java -Xmx2G -cp " + path_to_Mate_tools + "transition-1.30.jar is2.lemmatizer.Lemmatizer"
cmd += " -model " + path_to_Mate_tools + "models/" + model_file
cmd += " -test " + tokenized
cmd += " -out " + lemmatized
with open(stderr_file, "a") as stderrfile:
  process = subprocess.Popen(shlex.split(cmd), stdout = stderrfile, stderr=stderrfile)
  (stdoutdata, stderrdata) = process.communicate()

###################
# tag and parse
print "INFO: tag and parse"
model_file = "pet-ger-S2a-40-0.25-0.1-2-2-ht4-hm4-kk0"
cmd = "java -Xmx3G -cp " + path_to_Mate_tools + "transition-1.30.jar is2.transitionS2a.Parser"
cmd += " -model " + path_to_Mate_tools + "models/" + model_file
cmd += " -test " + lemmatized
cmd += " -out " + parsed
with open(stderr_file, "a") as stderrfile:
  process = subprocess.Popen(shlex.split(cmd), stdout = stderrfile, stderr=stderrfile)
  (stdoutdata, stderrdata) = process.communicate()

###################
# transform to conll06
# - do the same as PropsDE
with open(parsed) as conll09file:
    with open(parsed_conll06, "w") as conll06file:
        for line in conll09file:
            columns = line.strip().split("\t")
            if len(columns) != 14:
                conll06file.write(line)
            else:
                new_columns = columns[0:2] + columns[3:4] + columns[5:6] + columns[5:7] + columns[9:10] + columns[11:14]
                new_line = "\t".join(new_columns) + "\n"
                conll06file.write(new_line)

###################
# collapse dependencies
print "INFO: collapse"
cmd = "java -jar " + path_to_JobImText + "org.jobimtext.collapsing.jar"
cmd += " -i " + parsed_conll06 # input file
cmd += " -o " + collapsed_path # output file
cmd += " -sf" # process single file instead of folder
cmd += " -l de" # german language
cmd += " -r " + collapsing_rules_file #rule file
cmd += " -f c" #input file format is conll
cmd += " -np" # no parsing -- already parsed
cmd += " -nt" # no tagging -- already tagged
#cmd += " -depout" # write stanford dependencies style output instead of conll
#cmd += " -addpos" # write POS tags to dependency output
with open(stderr_file, "a") as stderrfile:
  process = subprocess.Popen(shlex.split(cmd), stdout = stderrfile, stderr=stderrfile)
  (stdoutdata, stderrdata) = process.communicate()

print "INFO: extract"
cmd = "python train.py extract configs/extract_3x1024_GRU_bidirectional_distinct_emb128d.ini " + collapsed
subprocess.call(shlex.split(cmd))
