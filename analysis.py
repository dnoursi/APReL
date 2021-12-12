

import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

###
#
# Analysis of the learned reward function from user studies:
# read all data, conduct unequal variance t-test, compute dof
# https://www.socscistatistics.com/pvalues/tdistribution.aspx
# 
###

fnames = ["studyno1_multi.pkl", "studyno2_multi.pkl", "studyno1_single.pkl", "studyno2_single.pkl", "studyno3_single.pkl"]

singles = []
multis = []
for i,fname in enumerate(fnames):
    infile = open("datafiles/"+fname, "rb")
    inpkl = pkl.load(infile)
    inweights = inpkl["weightss"]
    if i < 2:
        multis.append(inweights[-1][1])
    else:
        singles.append(inweights[-1][1])

singles = np.array(singles)
multis = np.array(multis)

t = (multis.mean() - singles.mean()) / np.sqrt(singles.var()/len(singles) + multis.var()/len(multis))
r1 = singles.var()**2 / len(singles)
r2 = multis.var()**2 / len(multis)
dof = (r1+r2)**2 / (r1**2 / (len(singles)-1) + r2**2 / (len(multis)-1))
print(t)
print(dof)

bin_centers = ["Algorithm 1", "Algorithm 2"]
y=[singles.mean(), multis.mean()]
yerr = [singles.std(), multis.std()]
plt.bar(bin_centers, y, yerr=yerr, capsize=6)
plt.title("Learned max_pos reward")
plt.ylabel("max_pos value")
plt.show()



###
#
# Analysis of the user responses from user studies:
# read all data, aggregate Likert item into Likert scale, 
# conduct unequal variance t-test, compute dof
# https://www.socscistatistics.com/pvalues/tdistribution.aspx
# 
###

# Three individuals took this survey
singles_nonrep = [2,3,1]
singles_nonrep=np.array(singles_nonrep)
singles_novel=[4,2,1]
singles_novel=np.array(singles_novel)
singles_int=[5,3,1]
singles_int=np.array(singles_int)
singles = np.mean([singles_nonrep, singles_novel,singles_int], axis=0)

# Two groups of two individuals took this survey
multis_nonrep=[3,2,3,2]
multis_nonrep=np.array(multis_nonrep)
multis_novel = [3,4,3,2]
multis_novel=np.array(multis_novel)
multis_int = [4,4,4,2]
multis_int=np.array(multis_int)
multis = np.mean([multis_nonrep, multis_novel,multis_int],axis=0)

bin_centers = ["Algorithm 1", "Algorithm 2"]
y=[singles.mean(), multis.mean()]
yerr = [singles.std(), multis.std()]
plt.bar(bin_centers, y, yerr=yerr, capsize=6)
plt.title("Likert scale responses for novelty")
plt.ylabel("Response value")
plt.show()

t = (multis.mean() - singles.mean()) / np.sqrt(singles.var()/len(singles) + multis.var()/len(multis))
r1 = singles.var()**2 / len(singles)
r2 = multis.var()**2 / len(multis)
dof = (r1+r2)**2 / (r1**2 / (len(singles)-1) + r2**2 / (len(multis)-1))
print(t)
print(dof)

# This is individual likert item analysis, which is incorrect
if False:
    bin_centers = ["Interesting", "Novel", "Non-Repetitive"]
    y=[singles_int.mean(), singles_novel.mean(), singles_nonrep.mean()]
    yerr=[singles_int.std(), singles_novel.std(), singles_nonrep.std()]
    plt.bar(bin_centers, y, yerr=yerr, capsize=6)
    plt.title("Mean and variance of survey results for Algorithm 1")
    #plt.ylabel("max_pos value")
    plt.show()


    bin_centers = ["Interesting", "Novel", "Non-Repetitive"]
    y=[multis_int.mean(), multis_novel.mean(), multis_nonrep.mean()]
    yerr=[multis_int.std(), multis_novel.std(), multis_nonrep.std()]
    plt.bar(bin_centers, y, yerr=yerr, capsize=6)
    plt.title("Mean and variance of survey results for Algorithm 2")
    #plt.ylabel("max_pos value")
    plt.show()




