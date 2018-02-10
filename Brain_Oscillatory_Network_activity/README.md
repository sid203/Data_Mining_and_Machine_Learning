########################################################################################################################################################################

Brain Oscillatory and Network Analysis during resting states-----


Summary: 

This project carries out analysis of EEG data obtained from 64 channels for two rest conditions that is eyes open and eyes closed. As a result there are two files, each corresponding to either of the rest states. The analysis is described in detail in the jupyter notebook under sectional headings. It involves Spectral analysis using Power spectral density estimation using welch's method, constructing confidence intervals, p values etc. The connectivity among channels is also estimated through MVAR estimator using Direct Transfer function. In the subsequent sections, Motif analysis and Community detection using Louvain clustering were implemented. Along with the jupyter notebook, a report file also exists which gives a bried summary of each of the sections of the analysis. 

Description of files: 

1.  EEG_data_analysis.ipynb : single Jupyter notebook having the complete analysis.
2.  brain-oscillatory-network.pdf : Report file explaining corresponding methodologies.
3.  data : Folder containing the input datasets and relevant intermediate data generated during the analysis. 
4. figures: plots and figures of important conclusions. 

