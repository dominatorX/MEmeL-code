We provide the full code for rainfall precipitation with ConvGRU and MEmeL.

Please first download dataset via their official site[https://github.com/sxjscience/HKO-7#download-the-hko-7-dataset-and-use-the-iterator].

Configurations are set in "now/config.py", please fill in the paths. Set "__C.GLOBAL.EMEL" to True to enable MEmeL.

For finetuning, go to the directory "exp" and run "python main.py".