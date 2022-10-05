# Competitive Evolution
This repository contains the source code which can be used to replicate the experiments reported in the article of Stefano Nolfi and Paolo Pagliuca entitled "Global Progress in Competitive Co-Evolution: A Systematic Comparison of Alternative Methods"

The experiments can be replicated by using the evorobotpy2 library available from https://github.com/snolfi/evorobotpy2 extended with the source files included in this repository. The instruction for using the evorobotpy2 library are included in Section 13.5 and 13.7 of the open access book available from https://bacrobotics.com/. To extend the library you should: (1) overwrite the file evorobotpy2/bin/es.py with the file es.py included in this repository, (2) copy the files archive.py, archivestar.py maxsolvestar.py and generalist.py, that contain the implementation of the corresponding algorithms, in the ./bin directory of the evorobotpy2 folder, and (3) copy the ./xarchive, ./xarchivestar, ./xmaxsolvestar and ./xgeneralist folders, which contain the .ini files with the setting of the hyperparameters, in the ./evorobotpt2 folder and run the corresponding experiments from these folders. 
